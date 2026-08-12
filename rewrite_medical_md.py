#!/usr/bin/env python3
"""
rewrite_medical_md.py
=====================

Convert medical textbook chapters (imperfectly extracted from PDF) into tagged
markdown study documents for the knowledge-graph pipeline.

The governing rule is faithfulness. A textbook chapter is already finished
prose, so the body text is carried over rather than rewritten: every content
paragraph in the source reaches a ``<con>`` in the output, with only the
mechanical repairs the PDF extractor makes necessary. The document metadata is
the one place inference is allowed, because the source has no counterpart for
it.

An earlier version of this tool targeted slide-derived markdown, where the
source was a keyword dump with no connected prose. There, generation was the
only option and most of the machinery existed to keep generated prose honest.
None of that applies to a textbook, and none of it survives.

Pipeline
--------
  1. normalise  -- repair the PDF extractor's damage (see below)
  2. parse      -- scan into a flat element stream in source order
  3. metadata   -- title/objectives/summary/glossary from the source;
                   one LLM call for <topic>, which has no source counterpart
  4. link       -- turn in-document 'Figure 2-1' mentions into <figref>
  5. render     -- assemble the tagged markdown
  6. verify     -- prove every source paragraph reached the output

Extraction damage repaired in stage 1
-------------------------------------
  * U+00A0 is used as the word separator throughout;
  * U+00AD stands in for a real hyphen ('Henderson<AD>Hasselbalch');
  * paragraphs are split mid-sentence wherever a page break fell.

Backend
-------
An OpenAI-compatible vLLM server:
    base url : http://jupyter-02.aml1.id.iosda.org:8888
    model    : huatuogpt-o1-72b-4bit   (mlx-community/HuatuoGPT-o1-72B-4bit)

HuatuoGPT-o1 is a reasoning model; its "## Thinking / ## Final Response"
scaffolding (and any <think> block) is stripped from every completion.

Output tag vocabulary
---------------------
See TAG_VOCABULARY below and the table printed by ``--print-schema``.

Usage
-----
    python rewrite_medical_md.py chapter.md -o out.md
    python rewrite_medical_md.py chapter.md -o out.md --offline   # no network
    python rewrite_medical_md.py chapter.md -o out.md -v --cache-dir ./cache
    python rewrite_medical_md.py --print-schema

Zero third-party dependencies (stdlib only).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import textwrap
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://jupyter-02.aml1.id.iosda.org:8888"
DEFAULT_MODEL = "huatuogpt-o1-72b-4bit"
MODEL_REPO = "mlx-community/HuatuoGPT-o1-72B-4bit"

DEFAULT_TIMEOUT = 900          # seconds; a 72B 4-bit MLX server is not fast
DEFAULT_RETRIES = 4
DEFAULT_WORKERS = 4

MAXTOK_TOPIC = 1024
MAXTOK_CAPTION = 1024
MAXTOK_OBJECTIVES = 2048
MAXTOK_SUMMARY = 2048

TEMP_TOPIC = 0.30
TEMP_CAPTION = 0.15            # conservative: stay close to the source text
TEMP_OBJECTIVES = 0.35
TEMP_SUMMARY = 0.35

VERBOSE = False


def log(msg: str) -> None:
    """Write a progress line to stderr, but only under --verbose."""
    if VERBOSE:
        sys.stderr.write("[rewrite] " + msg + "\n")
        sys.stderr.flush()


# --------------------------------------------------------------------------
# Tag vocabulary (documented, comprehensive)
# --------------------------------------------------------------------------

# (tag, where it appears, what it holds)
TAG_VOCABULARY: List[Tuple[str, str, str]] = [
    ("doc", "root", "Wraps the entire generated document."),
    ("meta", "doc", "Document-level metadata block."),
    ("title", "meta", "Chapter title. Plain text, no '#' marker."),
    ("topic", "meta", "One-line statement of the subject matter. The only "
                      "element inferred rather than carried over."),
    ("src", "meta", "Provenance: source filename, book, figure and table "
                    "counts, model, timestamp."),
    ("obj", "doc", "Learning objectives, from the chapter's OBJECTIVES list."),
    ("goal", "obj", "One learning objective. One per element, no bullet "
                    "marker."),
    ("sec", "doc", "One section. Attributes: id, level. Sections are flat; "
                   "depth is carried by the level attribute."),
    ("head", "sec", "Section heading, verbatim from the source."),
    ("con", "sec", "A content paragraph carried over from the source. Only "
                   "three things are changed: extraction artefacts are "
                   "repaired, page-break splits are rejoined, and in-document "
                   "figure and table mentions are wrapped in <figref>/<tblref>."),
    ("fig", "sec", "A figure, emitted at the position it occupied in the "
                   "source. Attributes: id; label when the source numbered "
                   "it; src when it carried an image file; panel for the "
                   "'A'/'B' letter on an unnumbered inline diagram."),
    ("cap", "fig / tbl", "Caption, taken from the source text."),
    ("desc", "fig", "Figure description. Emitted empty and reserved: nothing "
                    "in a textbook chapter describes the image itself."),
    ("figref", "con", "A reference to a <fig> in this document. Attribute: "
                      "id. Wraps the original mention text, so the prose "
                      "reads unchanged with the tags removed."),
    ("tbl", "sec", "A table, emitted at its source position. Attributes: id, "
                   "label. Holds <cap> and the source table markup."),
    ("tblref", "con", "A reference to a <tbl> in this document. Attribute: "
                      "id. Wraps the original mention text."),
    ("sum", "doc", "Chapter summary, from the source SUMMARY section."),
    ("def", "doc", "Glossary, from the source GLOSSARY section."),
    ("term", "def", "One definition, phrased as a complete sentence that "
                    "begins with the term (e.g. 'Bioethics is the area ...')."),
    ("ref", "doc", "Reference list, from the source RECOMMENDED READING."),
    ("cit", "ref", "A single citation."),
]

SCHEMA_NOTES = """
FAITHFULNESS

The chapter is a finished text, so <con> is carried over, not rewritten.
Every content paragraph in the source reaches exactly one <con>, and the
tool warns loudly rather than silently dropping one. The only edits applied
to carried-over text are mechanical repairs to PDF extraction damage:

  * U+00A0 word separators become ordinary spaces;
  * U+00AD, which the extractor emits in place of a real hyphen, becomes
    an ASCII '-';
  * paragraphs split mid-sentence at a page break are rejoined.

MARKDOWN SYNTAX POLICY

The tags carry the document structure, so structural markdown is not
emitted:

  * <title> and <head> hold plain text, with no '#' or '##' markers.
    Heading depth is expressed by the level attribute on <sec>.
  * List items are elements (<goal>, <term>, <cit>), not '- ' bullets.

Content-level markup inside <con>, <cap> and <tbl> is preserved exactly as
the source had it. That includes inline emphasis, <sup> and <sub>, LaTeX
spans such as $\\mathsf{pK_a}$, the HTML <table> blobs the extractor
produces, and any mermaid diagram carried inside a <fig>. Nothing in
element content is entity-escaped, because escaping would corrupt that
markup; only attribute values are escaped.

CROSS-REFERENCES

A mention of a figure or table defined in this chapter is wrapped in place:

    ... clinical insights (<figref id="fig-1-1">Figure 1-1</figref>).

Mentions that point outside the chapter, such as 'see Figure 40-5', are
left as plain text, because there is nothing in this document to link to.
Removing the tags restores the source sentence exactly.

BLOCK SHAPES

    <fig id="fig-2-1" label="Figure 2-1" src="images/a1b2c3.jpg">
    <cap>The water molecule has tetrahedral geometry.</cap>
    <desc></desc>
    </fig>

    <tbl id="tbl-2-1" label="Table 2-1">
    <cap>Bond Energies for Atoms of Biologic Significance</cap>
    <table>...</table>
    </tbl>
"""

SCHEMA_WIDTH = 78


def print_schema() -> None:
    """Print the tag vocabulary and the conventions that govern the output."""
    w1 = max(len(t) for t, _, _ in TAG_VOCABULARY) + 3
    w2 = max(len(p) for _, p, _ in TAG_VOCABULARY) + 2
    body = SCHEMA_WIDTH - w1 - w2

    print("OUTPUT TAG VOCABULARY")
    print()
    print(f"{'TAG'.ljust(w1)}{'PARENT'.ljust(w2)}MEANING")
    print()
    for tag, parent, meaning in TAG_VOCABULARY:
        lines = textwrap.wrap(meaning, body) or [""]
        print(f"{('<' + tag + '>').ljust(w1)}{parent.ljust(w2)}{lines[0]}")
        for line in lines[1:]:
            print(" " * (w1 + w2) + line)
    print(SCHEMA_NOTES)


# --------------------------------------------------------------------------
# LLM client (OpenAI-compatible /v1/chat/completions)
# --------------------------------------------------------------------------

class LLMError(RuntimeError):
    """Raised when the backend is unreachable or answers with junk."""


class LLMClient:
    """Minimal OpenAI-compatible chat client with retries and an on-disk cache."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        cache_dir: Optional[str] = None,
        debug_dir: Optional[str] = None,
    ) -> None:
        """Configure the backend.

        Creates cache_dir and debug_dir if given. Neither is required; both
        default to off.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.cache_dir = cache_dir
        self.debug_dir = debug_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

    # -- endpoint -------------------------------------------------------
    def _endpoint(self) -> str:
        """Return the chat-completions URL, tolerating a base url that already
        ends in '/v1'.
        """
        base = self.base_url
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    # -- cache ----------------------------------------------------------
    def _cache_path(self, payload: Dict[str, Any]) -> Optional[str]:
        """Return this request's cache file, or None when caching is off.

        The key is a hash of the whole payload, so changing the prompt, the
        model or the sampling parameters misses the cache, as it should.
        """
        if not self.cache_dir:
            return None
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return os.path.join(self.cache_dir, hashlib.sha256(blob).hexdigest()[:32] + ".json")

    # -- public ---------------------------------------------------------
    def chat(self, system: str, user: str, max_tokens: int = 2048,
             temperature: float = 0.4, tag: str = "") -> str:
        """Return just the model's answer. See chat_full for the parameters."""
        return self.chat_full(system, user, max_tokens, temperature, tag)[0]

    def chat_full(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.4,
        tag: str = "",
    ) -> Tuple[str, str]:
        """Return (final_answer, thinking_block).

        The thinking block is kept because reasoning models sometimes draft the
        real answer inside it and then emit an empty or skeletal final response.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stream": False,
        }

        cpath = self._cache_path(payload)
        if cpath and os.path.exists(cpath):
            log(f"cache hit  [{tag}]")
            with open(cpath, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
            return blob["text"], blob.get("thinking", "")

        raw = self._post_with_retries(payload, tag)
        self._dump(tag, user, raw)
        final, thinking = split_reasoning(raw)

        if cpath:
            with open(cpath, "w", encoding="utf-8") as fh:
                json.dump({"text": final, "thinking": thinking}, fh, ensure_ascii=False)
        return final, thinking

    def _dump(self, tag: str, prompt: str, raw: str) -> None:
        """Write the prompt and the unprocessed completion under --debug-dir.

        Best-effort: a failed dump is logged and swallowed, because losing a
        diagnostic file must not lose the run.
        """
        if not self.debug_dir:
            return
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", tag or "untagged")
        path = os.path.join(self.debug_dir, f"{safe}.txt")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("=== PROMPT ===\n" + prompt +
                         "\n\n=== RAW COMPLETION ===\n" + raw + "\n")
        except OSError as exc:                            # noqa: BLE001
            log(f"debug dump failed for {tag}: {exc}")

    # -- transport ------------------------------------------------------
    def _post_with_retries(self, payload: Dict[str, Any], tag: str) -> str:
        """POST with exponential backoff, capped at 30s between attempts.

        Raises LLMError once the attempt budget is exhausted.
        """
        last: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                t0 = time.time()
                out = self._post(payload)
                log(f"llm ok     [{tag}] {time.time() - t0:.1f}s  {len(out)} chars")
                return out
            except Exception as exc:                      # noqa: BLE001
                last = exc
                wait = min(2 ** attempt, 30)
                log(f"llm fail   [{tag}] attempt {attempt}/{self.retries}: {exc}; retry in {wait}s")
                if attempt < self.retries:
                    time.sleep(wait)
        raise LLMError(f"LLM request failed after {self.retries} attempts [{tag}]: {last}")

    def _post(self, payload: Dict[str, Any]) -> str:
        """Issue one request and return the message content.

        Reads the bearer token from VLLM_API_KEY, defaulting to 'EMPTY' for a
        server with authentication disabled. Raises LLMError if the response
        does not have the expected shape.
        """
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint(),
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + os.environ.get("VLLM_API_KEY", "EMPTY"),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {str(body)[:400]}") from exc


class OfflineClient(LLMClient):
    """Deterministic stand-in used by --offline; never touches the network.

    Inference is now confined to metadata, so a run with no model at all still
    produces a complete and correct document -- only <topic> degrades, from an
    inferred sentence to one derived from the title. That makes offline the
    normal way to work on the parser, not merely a test mode.
    """

    def __init__(self, doc: "SourceDoc") -> None:
        """Bind to the parsed document the derived answers are read from."""
        super().__init__(cache_dir=None)
        self.doc = doc

    def chat_full(self, system: str, user: str, max_tokens: int = 2048,
                  temperature: float = 0.4, tag: str = "") -> Tuple[str, str]:
        """Answer from the parsed document instead of the model.

        Only ``tag`` is consulted; the prompt and sampling parameters are
        accepted and ignored so call sites need no offline special case.
        """
        return self._derive(tag), ""

    def _derive(self, tag: str) -> str:
        """Build the answer for one prompt tag from the source document."""
        kind, _, arg = tag.partition(":")
        if kind == "topic":
            heads = [s.heading for s in self.doc.sections][:4]
            tail = "; ".join(heads)
            return f"{self.doc.title}." + (f" Covers {tail}." if tail else "")
        if kind == "objectives":
            return "\n".join(f"Understand {s.heading.lower()}."
                             for s in self.doc.sections[:8])
        if kind == "summary":
            return ""
        if kind == "caption":
            fig = next((f for f in self.doc.figures if f.fid == arg), None)
            return fig.label if fig else ""
        return ""


_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_OPEN_THINK_RE = re.compile(r"^.*?</think>", re.S | re.I)
_FINAL_RE = re.compile(r"##+\s*Final\s+Response\s*:?\s*", re.I)


def split_reasoning(text: str) -> Tuple[str, str]:
    """Separate a reasoning model's answer from its chain of thought.

    Returns (final_answer, thinking). Nothing is thrown away: callers can fall
    back to the thinking block when the final answer comes back empty, which
    reasoning models do occasionally under a multi-part output contract.
    """
    thinking_parts: List[str] = []

    for m in _THINK_RE.finditer(text):
        thinking_parts.append(m.group(0))
    text = _THINK_RE.sub("", text)

    if "</think>" in text.lower():
        m = _OPEN_THINK_RE.match(text)
        if m:
            thinking_parts.append(m.group(0))
        text = _OPEN_THINK_RE.sub("", text, count=1)

    parts = _FINAL_RE.split(text)
    if len(parts) > 1:
        thinking_parts.append("".join(parts[:-1]))
        text = parts[-1]

    text = re.sub(r"^\s*##+\s*Thinking\s*:?\s*$", "", text, flags=re.M | re.I)
    thinking = re.sub(r"</?think>", "", "\n".join(thinking_parts), flags=re.I)
    thinking = re.sub(r"^\s*##+\s*Thinking\s*:?\s*$", "", thinking, flags=re.M | re.I)
    return text.strip(), thinking.strip()


# --------------------------------------------------------------------------
# Stage 1: normalising the extractor's damage
# --------------------------------------------------------------------------

# The private-use area is where the extractor parks glyphs it could not map.
_PUA_RE = re.compile("[\\ue000-\\uf8ff]")
_EMPTY_TAG_RE = re.compile(r"<(sup|sub)>\s*</\1>", re.I)

# Kept atomic when splitting into blocks: both span blank lines, and both are
# meaningless in pieces.
_ATOMIC_RE = re.compile(r"<(details|table)\b.*?</\1>", re.S | re.I)
_NL_HOLD = "\x01"

_SENTENCE_END_RE = re.compile(r"[.!?:;][\"'’”)\]]?$")


def normalize_text(text: str) -> str:
    """Undo the character-level damage the PDF extractor leaves behind.

    U+00A0 is used as the word separator throughout the corpus and U+00AD
    stands in for a real hyphen, so both have to go before anything tries to
    tokenise or pattern-match the text.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u00ad", "-")
    text = _PUA_RE.sub("", text)
    text = _EMPTY_TAG_RE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text


def split_blocks(text: str) -> List[str]:
    """Split on blank lines, holding <details> and <table> regions together."""
    held = _ATOMIC_RE.sub(lambda m: m.group(0).replace("\n", _NL_HOLD), text)
    parts = re.split(r"\n\s*\n", held)
    return [p.replace(_NL_HOLD, "\n").strip() for p in parts if p.strip()]


def join_page_breaks(blocks: Sequence[str]) -> List[str]:
    """Rejoin paragraphs the extractor split mid-sentence at a page break.

    The signature is unambiguous in this corpus: the first fragment stops
    without sentence-final punctuation and the second resumes in lower case.
    Requiring both conditions is what keeps genuinely separate paragraphs, and
    the alt-text lines that follow a caption, from being glued together.
    """
    out: List[str] = []
    for block in blocks:
        if out and _is_continuation(out[-1], block):
            out[-1] = out[-1].rstrip() + " " + block.lstrip()
        else:
            out.append(block)
    return out


def _is_continuation(prev: str, nxt: str) -> bool:
    """Report whether nxt is the tail of a paragraph broken across a page."""
    if _is_structural(prev) or _is_structural(nxt):
        return False
    if _SENTENCE_END_RE.search(prev.rstrip()):
        return False
    return bool(re.match(r"[a-z(]", nxt))


def _is_structural(block: str) -> bool:
    """Report whether a block is markup rather than prose, and so unjoinable."""
    head = block.lstrip()
    if head.startswith(("#", "!", "<", "|", ">")):
        return True
    if CREDIT_RE.match(head) or FIGURE_HEAD_RE.match(head):
        return True
    return bool(TABLE_HEAD_RE.match(head))


# --------------------------------------------------------------------------
# Stage 2: source parsing
# --------------------------------------------------------------------------

_NUM = r"(\d+)\s*[–—-]\s*(\d+)"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
FIGURE_HEAD_RE = re.compile(r"^(?:#{1,6}\s+)?FIGURES?\s+" + _NUM + r"\s*(.*)$", re.S)
TABLE_HEAD_RE = re.compile(r"^(?:#{1,6}\s+)?TABLES?\s+" + _NUM + r"\s*(.*)$", re.S)
IMAGE_RE = re.compile(r"^!\[[^\]]*\]\(([^)]+)\)\s*(.*)$", re.S)
DETAILS_RE = re.compile(r"^<details\b", re.I)
TABLE_BODY_RE = re.compile(r"^<table\b", re.I)
CREDIT_RE = re.compile(r"^Source:\s", re.I)
CHAPTER_RE = re.compile(r"^Chapter\s+(\d+)\s*[:.–—-]\s*(.+)$", re.I)

# The chapters name their fixed sections in a stable way; each is handled by a
# dedicated collector rather than becoming an ordinary <sec>.
SPECIAL_HEADINGS = {
    "objectives": "objectives",
    "summary": "summary",
    "glossary": "glossary",
    "recommended reading": "references",
    "references": "references",
    "further reading": "references",
}

# The machine-written alt text that trails a caption. Anchored on an opening
# determiner and required to reach a depiction verb without crossing a full
# stop, so a real caption or a body paragraph cannot match by accident.
ALT_TEXT_RE = re.compile(
    r"""^(?:An?|The|Two|Three|Four|Five|Six|Several|Illustrations?|Images?
          |Diagrams?|Chemical|Photographs?|Micrographs?|Graphs?|Schematics?
          |Models?|Flowcharts?|Charts?|Tables?)\b
        [^.]{0,200}?
        \b(?:illustrat\w*|depict\w*|show\w*|display\w*|diagram\w*|graph\w*
            |plots?|plotted|photograph\w*|micrograph\w*|images?|models?
            |represent\w*|reads?|marked|labell?ed|visuali[sz]\w*)\b
    """,
    re.X | re.I | re.S,
)


@dataclass
class Figure:
    """A numbered figure, or an unnumbered inline image, from the source."""

    fid: str
    label: str
    order: int
    caption: str = ""
    image_src: str = ""
    extra: str = ""
    # 'A'/'B' on an unnumbered image: the source's only handle on that panel.
    panel: str = ""
    # Parsed but deliberately not rendered: <desc> is empty for now, and this
    # is the text it would be built from.
    alt_text: str = ""


@dataclass
class Table:
    """A numbered table and the source markup of its body."""

    tid: str
    label: str
    order: int
    caption: str = ""
    body: str = ""


@dataclass
class Element:
    """One item in a section's body, in source order.

    ``kind`` is 'para', 'figure' or 'table'. Exactly one of the three payload
    fields is set.
    """

    kind: str
    text: str = ""
    figure: Optional[Figure] = None
    table: Optional[Table] = None


@dataclass
class Section:
    """A source heading and everything that followed it."""

    sid: str
    heading: str
    level: int
    elements: List[Element] = field(default_factory=list)


@dataclass
class Block:
    """A source block plus the disposition the scanner gave it.

    The disposition is what makes the content-preservation check in ``verify``
    possible: every block is accounted for by name, so a paragraph the scanner
    misreads shows up as a warning instead of vanishing.
    """

    text: str
    disposition: str = "unclaimed"


@dataclass
class SourceDoc:
    """Everything parsed out of one chapter file."""

    path: str
    book: str
    title: str
    author: str
    sections: List[Section]
    figures: List[Figure]
    tables: List[Table]
    objectives: List[str]
    summary: List[str]
    glossary: List[str]
    references: List[str]
    blocks: List[Block]

    @property
    def content_words(self) -> int:
        """Total words across every carried-over content paragraph."""
        return sum(word_count(e.text)
                   for s in self.sections for e in s.elements
                   if e.kind == "para")


def parse_source(path: str) -> SourceDoc:
    """Scan a chapter file into sections, figures, tables and fixed lists."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()

    blocks = [Block(t) for t in join_page_breaks(split_blocks(normalize_text(raw)))]
    scanner = _Scanner(path, blocks)
    return scanner.run()


class _Scanner:
    """Single forward pass over the block list.

    Kept as a class only because the pass is stateful: the current section, the
    current fixed-section collector, and the figure/table counters all have to
    persist across blocks, and threading them through free functions was worse.
    """

    def __init__(self, path: str, blocks: List[Block]) -> None:
        """Prepare a pass over ``blocks``, which the scan annotates in place."""
        self.path = path
        self.blocks = blocks
        self.pos = 0
        self.book = ""
        self.title = ""
        self.author = ""
        self.sections: List[Section] = []
        self.figures: List[Figure] = []
        self.tables: List[Table] = []
        self.objectives: List[str] = []
        self.summary: List[str] = []
        self.glossary: List[str] = []
        self.references: List[str] = []
        self.mode = "preamble"
        self.section: Optional[Section] = None
        self.used_ids: Dict[str, int] = {}

    # -- driver ---------------------------------------------------------
    def run(self) -> SourceDoc:
        """Consume every block and return the assembled document."""
        while self.pos < len(self.blocks):
            start = self.pos
            self._step()
            if self.pos == start:                 # never stall
                self.pos += 1
        if not self.title:
            self.title = self._fallback_title()
        doc = SourceDoc(
            path=self.path, book=self.book, title=self.title, author=self.author,
            sections=[s for s in self.sections if s.elements],
            figures=self.figures, tables=self.tables,
            objectives=self.objectives, summary=self.summary,
            glossary=self.glossary, references=self.references,
            blocks=self.blocks,
        )
        log(f"parsed {os.path.basename(self.path)}: "
            f"{len(doc.sections)} sections, {len(doc.figures)} figures, "
            f"{len(doc.tables)} tables, {doc.content_words} content words")
        return doc

    def _step(self) -> None:
        """Classify the block at the cursor and consume it and anything it owns."""
        block = self.blocks[self.pos]
        text = block.text

        heading = HEADING_RE.match(text) if "\n" not in text else None
        if heading and not FIGURE_HEAD_RE.match(text) and not TABLE_HEAD_RE.match(text):
            self._open_heading(block, heading.group(2))
            return
        if FIGURE_HEAD_RE.match(text):
            self._read_figure(block)
            return
        if TABLE_HEAD_RE.match(text):
            self._read_table(block)
            return
        self._read_body(block)

    # -- headings -------------------------------------------------------
    def _open_heading(self, block: Block, title: str) -> None:
        """Start a new section, or switch to a fixed-section collector."""
        self.pos += 1
        block.disposition = "heading"

        chapter = CHAPTER_RE.match(title)
        if chapter:
            self.title = chapter.group(2).strip()
            self.mode = "chapter"
            self.section = None
            return

        special = SPECIAL_HEADINGS.get(title.strip().lower().rstrip(":"))
        if special:
            self.mode = special
            self.section = None
            return

        self.mode = "body"
        self.section = Section(self._section_id(title), title,
                               _heading_level(title))
        self.sections.append(self.section)

    def _section_id(self, title: str) -> str:
        """Slugify a heading, suffixing a counter if the slug repeats."""
        base = re.sub(r"[^a-z0-9]+", "-", clean_inline_md(title).lower()).strip("-")
        if len(base) > 60:                     # trim on a word boundary
            base = base[:60].rsplit("-", 1)[0]
        base = base or "section"
        seen = self.used_ids.get(base, 0)
        self.used_ids[base] = seen + 1
        return base if not seen else f"{base}-{seen + 1}"

    # -- figures --------------------------------------------------------
    def _read_figure(self, block: Block) -> None:
        """Consume a numbered figure and emit it at this position."""
        m = FIGURE_HEAD_RE.match(block.text)
        assert m is not None
        self.pos += 1
        block.disposition = "figure-head"

        label = f"Figure {m.group(1)}-{m.group(2)}"
        fig = Figure(fid=f"fig-{m.group(1)}-{m.group(2)}", label=label,
                     order=len(self.figures), caption=m.group(3).strip())
        self._read_figure_body(fig)
        self.figures.append(fig)
        self._emit(Element("figure", figure=fig))

    def _read_figure_body(self, fig: Figure) -> None:
        """Consume the optional parts that follow a figure head, in order.

        Each part is recognised by shape, and the walk stops at the first block
        that is not one of them, so body prose is never absorbed into a figure.
        """
        if not fig.caption:
            nxt = self._peek()
            if nxt is not None and self._is_caption_block(nxt.text):
                fig.caption = clean_heading(nxt.text)
                nxt.disposition = "caption"
                self.pos += 1

        nxt = self._peek()
        if nxt is not None and _is_alt_text(nxt.text):
            fig.alt_text = nxt.text
            nxt.disposition = "figure-alt"
            self.pos += 1

        nxt = self._peek()
        if nxt is not None:
            image = IMAGE_RE.match(nxt.text)
            if image:
                fig.image_src = image.group(1).strip()
                nxt.disposition = "figure-image"
                self.pos += 1

        nxt = self._peek()
        if nxt is not None and DETAILS_RE.match(nxt.text):
            fig.extra = _unwrap_details(nxt.text)
            nxt.disposition = "figure-detail"
            self.pos += 1

        nxt = self._peek()
        if nxt is not None and CREDIT_RE.match(nxt.text):
            nxt.disposition = "credit"
            self.pos += 1

    def _is_caption_block(self, text: str) -> bool:
        """Report whether a block could be a figure caption."""
        if _is_alt_text(text):
            return False
        if IMAGE_RE.match(text) or DETAILS_RE.match(text):
            return False
        if CREDIT_RE.match(text) or TABLE_BODY_RE.match(text):
            return False
        return not FIGURE_HEAD_RE.match(text) and not TABLE_HEAD_RE.match(text)

    # -- tables ---------------------------------------------------------
    def _read_table(self, block: Block) -> None:
        """Consume a numbered table with its body and emit it at this position."""
        m = TABLE_HEAD_RE.match(block.text)
        assert m is not None
        self.pos += 1
        block.disposition = "table-head"

        label = f"Table {m.group(1)}-{m.group(2)}"
        tbl = Table(tid=f"tbl-{m.group(1)}-{m.group(2)}", label=label,
                    order=len(self.tables),
                    caption=clean_heading(m.group(3).strip()))

        nxt = self._peek()
        if nxt is not None and TABLE_BODY_RE.match(nxt.text):
            tbl.body = nxt.text.strip()
            nxt.disposition = "table-body"
            self.pos += 1

        nxt = self._peek()
        if nxt is not None and CREDIT_RE.match(nxt.text):
            nxt.disposition = "credit"
            self.pos += 1

        self.tables.append(tbl)
        self._emit(Element("table", table=tbl))

    # -- body -----------------------------------------------------------
    def _read_body(self, block: Block) -> None:
        """Route a non-heading, non-figure block by the collector in force."""
        self.pos += 1
        text = block.text

        if self.mode == "preamble" and not self.book:
            self.book = clean_inline_md(text)
            block.disposition = "book"
            return
        if self.mode == "chapter" and not self.author:
            self.author = clean_inline_md(text)
            block.disposition = "author"
            return
        if CREDIT_RE.match(text):
            block.disposition = "credit"
            return

        if self.mode == "objectives":
            self.objectives.extend(_list_items(text))
            block.disposition = "objectives"
            return
        if self.mode == "summary":
            self.summary.extend(_list_items(text))
            block.disposition = "summary"
            return
        if self.mode == "glossary":
            self.glossary.append(text)
            block.disposition = "glossary"
            return
        if self.mode == "references":
            self.references.extend(_list_items(text))
            block.disposition = "references"
            return

        image = IMAGE_RE.match(text)
        if image:
            self._read_loose_image(block, image)
            return
        if DETAILS_RE.match(text) or TABLE_BODY_RE.match(text):
            self._attach_to_last_figure(block)
            return

        block.disposition = "content"
        self._emit(Element("para", text=text))

    def _read_loose_image(self, block: Block, image: "re.Match[str]") -> None:
        """An image with no FIGURE head still gets a <fig>, with a coined id.

        Chapter 3 carries several of these: unnumbered structural diagrams
        dropped inline. They are real figures and the prose sometimes points at
        them ('as in A, in the following figure'), so discarding them would
        lose content.
        """
        block.disposition = "figure-image"
        order = len(self.figures)
        fig = Figure(fid=f"fig-x{order + 1}", label="", order=order,
                     image_src=image.group(1).strip(),
                     panel=clean_heading(image.group(2))[:8])
        self.figures.append(fig)
        self._emit(Element("figure", figure=fig))

    def _attach_to_last_figure(self, block: Block) -> None:
        """A <details> or <table> blob trailing a figure belongs to it."""
        target = self.figures[-1] if self.figures else None
        if target is not None and not target.extra:
            target.extra = (_unwrap_details(block.text)
                            if DETAILS_RE.match(block.text) else block.text.strip())
            block.disposition = "figure-detail"
            return
        block.disposition = "content"
        self._emit(Element("para", text=block.text))

    # -- shared ---------------------------------------------------------
    def _peek(self) -> Optional[Block]:
        """Return the block at the cursor without consuming it, or None at end."""
        return self.blocks[self.pos] if self.pos < len(self.blocks) else None

    def _emit(self, element: Element) -> None:
        """Append to the open section, opening a catch-all one if none exists."""
        if self.section is None:
            self.section = Section(self._section_id("overview"), "Overview", 1)
            self.sections.append(self.section)
        self.section.elements.append(element)

    def _fallback_title(self) -> str:
        """Derive a title for a chapter with no 'Chapter N:' heading."""
        for section in self.sections:
            if section.heading:
                return section.heading
        return os.path.splitext(os.path.basename(self.path))[0].replace("_", " ")


def _heading_level(title: str) -> int:
    """Infer depth from casing, because the source makes everything '##'.

    The chapters set major sections in full capitals and subsections in title
    case, which is the only depth signal the extraction preserved.
    """
    letters = [c for c in title if c.isalpha()]
    if not letters:
        return 2
    upper = sum(1 for c in letters if c.isupper())
    return 1 if upper / len(letters) >= 0.8 else 2


def _is_alt_text(text: str) -> bool:
    """Report whether a block is the extractor's machine description of an image."""
    return "\n" not in text.strip() and bool(ALT_TEXT_RE.match(text.strip()))


def _unwrap_details(text: str) -> str:
    """Strip the <details>/<summary> chrome, keeping the payload."""
    body = re.sub(r"</?details\b[^>]*>", "", text, flags=re.I)
    body = re.sub(r"<summary\b[^>]*>.*?</summary>", "", body, flags=re.I | re.S)
    return body.strip()


def _list_items(text: str) -> List[str]:
    """Split a fixed-section block into items.

    The extractor writes objectives, summary points and citations as one block
    of lines separated by a trailing double space, so a line is an item.
    """
    items: List[str] = []
    for line in text.splitlines():
        line = strip_bullet(line)
        if not line or line.lower().startswith("after studying this chapter"):
            continue
        items.append(line)
    return items


def clean_heading(text: str) -> str:
    """Drop a '##' marker and collapse whitespace, keeping inline markup."""
    return re.sub(r"\s+", " ", MD_HEADING_RE.sub("", text)).strip()


# --------------------------------------------------------------------------
# Generic text helpers
# --------------------------------------------------------------------------

# A real markdown bullet is a marker FOLLOWED BY WHITESPACE. Requiring the
# trailing space -- and refusing to treat the first '*' of '**bold**' as a
# marker -- is what stops '**term** - x' from being mangled into '*term** - x'.
BULLET_RE = re.compile(r"^\s*(?:[-+•‣▪]|\*(?!\*)|\d+[.)])\s+")

EMPHASIS_RE = re.compile(r"(\*{1,3}|_{2,3})(?=\S)(.+?)(?<=\S)\1", re.S)
MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)


def strip_bullet(line: str) -> str:
    """Remove a leading list marker, leaving the item text."""
    return BULLET_RE.sub("", line).strip()


def clean_inline_md(text: str, keep_emphasis: bool = False) -> str:
    """Drop structural markdown; optionally keep inline emphasis."""
    text = MD_HEADING_RE.sub("", text)
    if not keep_emphasis:
        prev = None
        while prev != text:                    # handles ***nested*** cases
            prev = text
            text = EMPHASIS_RE.sub(r"\2", text)
    return text.strip()


DEF_SEP_RE = re.compile(r"^(.{1,90}?)\s*(?:—|–|--|::|:)\s+(.+)$", re.S)
_COPULA_RE = re.compile(r"^(is|are|refers?\b|denotes?\b|describes?\b|means?\b)", re.I)


def normalize_definition(line: str, keep_emphasis: bool = False) -> str:
    """Render a definition as a sentence: 'Bioethics is the area of ...'.

    Accepts either a full sentence (passed through) or the source glossary's
    'Term: definition' shape, which is converted.
    """
    s = clean_inline_md(strip_bullet(line), keep_emphasis)
    m = DEF_SEP_RE.match(s)
    if m:
        term, body = m.group(1).strip(), m.group(2).strip()
        # Only de-capitalise a leading article; leave acronyms and proper
        # nouns ('HGP: Human Genome Project ...') alone.
        if re.match(r"^(A|An|The)\s", body):
            body = body[0].lower() + body[1:]
        if not _COPULA_RE.match(body):
            body = "is " + body
        s = f"{term} {body}"
    s = re.sub(r"\s+", " ", s).strip()
    if s:
        s = s[0].upper() + s[1:]        # it is a sentence, so sentence-case it
    if s and not s.endswith((".", "!", "?")):
        s += "."
    return s


def word_count(text: str) -> int:
    """Count words, ignoring markdown punctuation that would inflate the total."""
    return len(re.sub(r"[#*`>|\[\]]", " ", text).split())


def extract_json(text: str) -> Any:
    """Pull the first balanced JSON object or array out of a model response."""
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.M)
    starts = [i for i, ch in enumerate(text) if ch in "{["]
    for start in starts:
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("no parseable JSON found in model response")


def escape_attr(value: str) -> str:
    """Escape an attribute value.

    Element content is never escaped, because <con>, <cap> and <tbl> carry the
    source's own markup and escaping would corrupt it. Attribute values are the
    one place a stray quote or ampersand would break a parser, so they are the
    one place escaping is applied.
    """
    return (value.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;").replace('"', "&quot;"))


def _run_parallel(fn, items: Sequence[Any], workers: int, label: str) -> None:
    """Apply fn to every item, on a thread pool when workers exceeds one.

    Exceptions propagate from the first future that raised, so a genuine
    failure is not hidden by concurrency.
    """
    items = list(items)
    if not items:
        return
    if workers <= 1:
        for it in items:
            fn(it)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, it) for it in items]
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            fut.result()
            done += 1
            log(f"{label}: {done}/{len(items)}")


# --------------------------------------------------------------------------
# Stage 3: metadata
# --------------------------------------------------------------------------

SYS_ANALYST = (
    "You read chapters of a medical textbook and answer with plain text only. "
    "No preamble, no markdown headings, no bullet markers, no commentary on "
    "the question. Answer with the requested text and nothing else."
)


def _chapter_digest(doc: SourceDoc, max_chars: int = 6000) -> str:
    """A compact view of the chapter for the metadata prompts."""
    parts = [f"TITLE: {doc.title}"]
    for section in doc.sections:
        parts.append(f"\n## {section.heading}")
        for element in section.elements:
            if element.kind == "para":
                parts.append(element.text)
                break
    digest = "\n".join(parts)
    return digest[:max_chars]


def prompt_topic(doc: SourceDoc) -> str:
    """Build the prompt asking for the chapter's one-line subject statement."""
    return (
        "Below is a digest of one chapter of a medical textbook.\n\n"
        f"{_chapter_digest(doc)}\n\n"
        "Write ONE sentence, at most 30 words, stating what this chapter is "
        "about. Name the actual subject matter, not the fact that it is a "
        "chapter. Do not begin with 'This chapter'. Output the sentence only."
    )


def prompt_objectives(doc: SourceDoc) -> str:
    """Build the prompt used only when a chapter has no OBJECTIVES section."""
    return (
        "Below is a digest of one chapter of a medical textbook.\n\n"
        f"{_chapter_digest(doc)}\n\n"
        "Write 5 to 8 learning objectives for this chapter, one per line. "
        "Each begins with a verb and states something a reader should be able "
        "to do after studying the chapter. No numbering, no bullet markers."
    )


def prompt_summary(doc: SourceDoc) -> str:
    """Build the prompt used only when a chapter has no SUMMARY section."""
    return (
        "Below is a digest of one chapter of a medical textbook.\n\n"
        f"{_chapter_digest(doc)}\n\n"
        "Write 5 to 8 summary points for this chapter, one per line. Each is "
        "a complete sentence stating a substantive conclusion from the "
        "chapter. No numbering, no bullet markers."
    )


def prompt_caption(fig: Figure, doc: SourceDoc) -> str:
    """Build the prompt for captioning a figure the source left uncaptioned."""
    return (
        f"In a chapter titled '{doc.title}', a figure labelled "
        f"'{fig.label or 'an unnumbered diagram'}' appears with no caption.\n\n"
        f"What the extractor recorded about the image:\n{fig.alt_text or fig.extra}\n\n"
        "Write ONE sentence, at most 25 words, that could serve as its "
        "caption. Describe only what the recorded text supports; invent "
        "nothing. Output the sentence only."
    )


def build_metadata(client: LLMClient, doc: SourceDoc, want_summary: bool) -> str:
    """Fill the metadata the source does not supply. Returns the topic line.

    Objectives, summary and glossary come from the chapter itself whenever it
    has them, which on this corpus is always; the model is called only for
    <topic>, which has no source counterpart, and to fill a genuine gap.
    """
    topic = client.chat(SYS_ANALYST, prompt_topic(doc),
                        max_tokens=MAXTOK_TOPIC, temperature=TEMP_TOPIC,
                        tag="topic").strip()
    topic = clean_heading(topic).split("\n")[0]

    if not doc.objectives:
        log("no OBJECTIVES section in source; inferring")
        raw = client.chat(SYS_ANALYST, prompt_objectives(doc),
                          max_tokens=MAXTOK_OBJECTIVES,
                          temperature=TEMP_OBJECTIVES, tag="objectives")
        doc.objectives = _list_items(raw)

    if want_summary and not doc.summary:
        log("no SUMMARY section in source; inferring")
        raw = client.chat(SYS_ANALYST, prompt_summary(doc),
                          max_tokens=MAXTOK_SUMMARY, temperature=TEMP_SUMMARY,
                          tag="summary")
        doc.summary = _list_items(raw)

    return topic or doc.title


def fill_missing_captions(client: LLMClient, doc: SourceDoc, workers: int) -> None:
    """Caption the figures the source left uncaptioned.

    A numbered figure almost always carries its caption in the source. The
    unnumbered inline diagrams do not, and those are the ones this reaches.
    """
    todo = [f for f in doc.figures if not f.caption and (f.alt_text or f.extra)]
    if not todo:
        return

    def one(fig: Figure) -> None:
        """Caption a single figure, logging and skipping on backend failure."""
        try:
            text = client.chat(SYS_ANALYST, prompt_caption(fig, doc),
                               max_tokens=MAXTOK_CAPTION,
                               temperature=TEMP_CAPTION, tag=f"caption:{fig.fid}")
        except LLMError as exc:                           # noqa: BLE001
            log(f"caption failed for {fig.fid}: {exc}")
            return
        fig.caption = clean_heading(text).split("\n")[0]

    _run_parallel(one, todo, workers, "captions")


# --------------------------------------------------------------------------
# Stage 4: cross-reference linking
# --------------------------------------------------------------------------

MENTION_RE = re.compile(
    r"\b(Figure|Figures|Table|Tables)\s+(\d+)\s*[–—-]\s*(\d+)([A-Z])?\b")


def link_references(text: str, known: Dict[str, str]) -> str:
    """Wrap in-document figure and table mentions in a reference tag.

    The mention text is kept and wrapped rather than replaced, so stripping the
    tags restores the source sentence exactly. Mentions that resolve to nothing
    in this chapter -- 'see Figure 40-5', which points at another chapter --
    are left alone, because there is no target to point at.
    """
    def replace(m: "re.Match[str]") -> str:
        """Wrap one mention, or return it untouched when it does not resolve."""
        kind = "fig" if m.group(1).lower().startswith("figure") else "tbl"
        ref_id = f"{kind}-{m.group(2)}-{m.group(3)}"
        if known.get(ref_id) != kind:
            return m.group(0)
        tag = "figref" if kind == "fig" else "tblref"
        return f'<{tag} id="{escape_attr(ref_id)}">{m.group(0)}</{tag}>'

    return MENTION_RE.sub(replace, text)


def reference_index(doc: SourceDoc) -> Dict[str, str]:
    """Map every figure and table id in this document to its kind."""
    index = {f.fid: "fig" for f in doc.figures}
    index.update({t.tid: "tbl" for t in doc.tables})
    return index


# --------------------------------------------------------------------------
# Stage 5: rendering
# --------------------------------------------------------------------------

def _caption(text: str, known: Dict[str, str]) -> str:
    """Flatten a caption to one line and link its cross-references."""
    return link_references(re.sub(r"\s+", " ", text).strip(), known)


def render_figure(fig: Figure, known: Dict[str, str]) -> str:
    """Render one <fig>. ``known`` links cross-references inside the caption."""
    attrs = f'id="{escape_attr(fig.fid)}"'
    if fig.label:
        attrs += f' label="{escape_attr(fig.label)}"'
    if fig.panel:
        attrs += f' panel="{escape_attr(fig.panel)}"'
    if fig.image_src:
        attrs += f' src="{escape_attr(fig.image_src)}"'
    out = [f"<fig {attrs}>"]
    out.append(f"<cap>{_caption(fig.caption, known)}</cap>")
    out.append("<desc></desc>")
    if fig.extra:
        out.append("")
        out.append(fig.extra)
    out.append("</fig>")
    return "\n".join(out)


def render_table(tbl: Table, known: Dict[str, str]) -> str:
    """Render one <tbl>. ``known`` links cross-references inside the caption."""
    attrs = f'id="{escape_attr(tbl.tid)}"'
    if tbl.label:
        attrs += f' label="{escape_attr(tbl.label)}"'
    out = [f"<tbl {attrs}>"]
    out.append(f"<cap>{_caption(tbl.caption, known)}</cap>")
    if tbl.body:
        out.append(tbl.body)
    out.append("</tbl>")
    return "\n".join(out)


def render_section(section: Section, known: Dict[str, str]) -> str:
    """Render one <sec>, walking its elements in source order."""
    out = [f'<sec id="{escape_attr(section.sid)}" level="{section.level}">',
           f"<head>{clean_heading(section.heading)}</head>", ""]
    for element in section.elements:
        if element.kind == "para":
            out.append("<con>")
            out.append(link_references(element.text, known))
            out.append("</con>")
        elif element.kind == "figure" and element.figure is not None:
            out.append(render_figure(element.figure, known))
        elif element.kind == "table" and element.table is not None:
            out.append(render_table(element.table, known))
        out.append("")
    out.append("</sec>")
    return "\n".join(out)


def render_document(doc: SourceDoc, topic: str, model: str,
                    want_summary: bool) -> str:
    """Assemble the whole tagged document.

    ``model`` is recorded in <src> for provenance, and ``want_summary``
    suppresses the <sum> block when --no-summary was given.
    """
    known = reference_index(doc)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    src = (f"source: {os.path.basename(doc.path)}"
           f" | book: {doc.book}" if doc.book else
           f"source: {os.path.basename(doc.path)}")
    src += (f" | figures: {len(doc.figures)} | tables: {len(doc.tables)}"
            f" | model: {model} | generated: {stamp}")

    out: List[str] = ["<doc>", "", "<meta>",
                      f"<title>{doc.title}</title>",
                      f"<topic>{topic}</topic>",
                      f"<src>{src}</src>",
                      "</meta>", ""]

    if doc.objectives:
        out.append("<obj>")
        for goal in doc.objectives:
            out.append(f"<goal>{goal}</goal>")
        out.append("</obj>")
        out.append("")

    for section in doc.sections:
        out.append(render_section(section, known))
        out.append("")

    if want_summary and doc.summary:
        out.append("<sum>")
        out.extend(doc.summary)
        out.append("</sum>")
        out.append("")

    if doc.glossary:
        out.append("<def>")
        for entry in doc.glossary:
            out.append(f"<term>{normalize_definition(entry, keep_emphasis=True)}</term>")
        out.append("</def>")
        out.append("")

    if doc.references:
        out.append("<ref>")
        for cit in doc.references:
            out.append(f"<cit>{cit}</cit>")
        out.append("</ref>")
        out.append("")

    out.append("</doc>")
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

# Dispositions that legitimately produce no <con> of their own, each for a
# reason that was decided rather than defaulted. A disposition missing from
# both this set and EMITTED_DISPOSITIONS is a bug, and verify says so.
SILENT_DISPOSITIONS = {
    "heading",        # becomes <head>, or selects a fixed-section collector
    "book",           # recorded in <src>
    "author",         # attribution already carried by the book line
    "credit",         # repeated per-figure copyright boilerplate
    "figure-alt",     # would populate <desc>, which is empty for now
    "figure-head", "figure-image", "figure-detail",
    "table-head", "table-body",
}

EMITTED_DISPOSITIONS = {
    "content", "caption", "objectives", "summary", "glossary", "references",
}

_TAG_RE = re.compile(r"</?[a-z][a-z0-9]*(?:\s[^>]*)?/?>", re.I)


def _plain(text: str) -> str:
    """Text with tags removed and whitespace collapsed, for comparison."""
    return re.sub(r"\s+", " ", _TAG_RE.sub("", text)).strip()


def verify(output: str, doc: SourceDoc) -> List[str]:
    """Check that nothing was lost and that every reference resolves."""
    problems: List[str] = []
    flat = _plain(output)

    known_dispositions = SILENT_DISPOSITIONS | EMITTED_DISPOSITIONS
    for block in doc.blocks:
        if block.disposition not in known_dispositions:
            problems.append(f"UNACCOUNTED source block ({block.disposition}): "
                            f"{block.text[:100]!r}")

    missing = 0
    for block in doc.blocks:
        if block.disposition != "content":
            continue
        if _plain(block.text) not in flat:
            missing += 1
            if missing <= 5:
                problems.append(f"CONTENT LOST: {block.text[:100]!r}")
    if missing > 5:
        problems.append(f"CONTENT LOST: and {missing - 5} more paragraph(s)")

    for fig in doc.figures:
        n = output.count(f'<fig id="{fig.fid}"')
        if n != 1:
            problems.append(f"figure {fig.fid} defined {n} times, expected 1")
    for tbl in doc.tables:
        n = output.count(f'<tbl id="{tbl.tid}"')
        if n != 1:
            problems.append(f"table {tbl.tid} defined {n} times, expected 1")

    known = reference_index(doc)
    for ref_id in re.findall(r'<(?:fig|tbl)ref id="([^"]+)"', output):
        if ref_id not in known:
            problems.append(f"dangling reference: {ref_id}")

    for tag in ("doc", "meta", "obj", "sum", "def", "ref"):
        if f"<{tag}>" in output and output.count(f"<{tag}>") != output.count(f"</{tag}>"):
            problems.append(f"unbalanced <{tag}> tags")
    for tag in ("sec", "fig", "tbl"):
        if output.count(f"<{tag} ") != output.count(f"</{tag}>"):
            problems.append(f"unbalanced <{tag}> tags")

    return problems


def disposition_report(doc: SourceDoc) -> str:
    """Summarise how many source blocks landed in each disposition."""
    counts: Dict[str, int] = {}
    for block in doc.blocks:
        counts[block.disposition] = counts.get(block.disposition, 0) + 1
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    """Convert one file end to end.

    Writes to args.output, or stdout when it is '-'. Returns 3 if the backend
    is unreachable, 2 if the input yielded no content, 1 if verification
    raised warnings, and 0 otherwise.
    """
    doc = parse_source(args.input)
    if not doc.sections:
        sys.stderr.write(f"error: no content found in {args.input}\n")
        return 2

    offline = args.offline or args.dry_run
    if offline:
        client: LLMClient = OfflineClient(doc)
        model = "offline"
    else:
        client = LLMClient(
            base_url=args.base_url, model=args.model, timeout=args.timeout,
            retries=args.retries, cache_dir=args.cache_dir,
            debug_dir=args.debug_dir,
        )
        model = args.model

    want_summary = not args.no_summary

    log("stage 1/3  metadata")
    try:
        topic = build_metadata(client, doc, want_summary)
    except LLMError as exc:                               # noqa: BLE001
        # Refusing here rather than falling back keeps a misconfigured backend
        # from quietly producing a document with degraded metadata.
        sys.stderr.write(f"error: {exc}\n")
        sys.stderr.write("hint: rerun with --offline to build the document "
                         "without the backend\n")
        return 3

    log("stage 2/3  captions")
    fill_missing_captions(client, doc, args.workers)

    log("stage 3/3  render")
    output = render_document(doc, topic, model, want_summary)

    problems = verify(output, doc)
    log("blocks: " + disposition_report(doc))
    for problem in problems:
        sys.stderr.write(f"warning: {problem}\n")

    if not args.output or args.output == "-":
        sys.stdout.write(output)
    else:
        parent = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(parent, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(output)
        print(f"wrote {args.output}  "
              f"({len(doc.sections)} sections, {len(doc.figures)} figures, "
              f"{len(doc.tables)} tables, {doc.content_words} content words, "
              f"{len(problems)} warning(s))")

    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    p = argparse.ArgumentParser(
        description="Convert medical textbook markdown into tagged study material.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"backend: {DEFAULT_BASE_URL}  model: {DEFAULT_MODEL} ({MODEL_REPO})",
    )
    p.add_argument("input", nargs="?", help="input markdown file")
    p.add_argument("-o", "--output", default="-",
                   help="output markdown file ('-' for stdout)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help="parallel LLM calls when captions have to be inferred")
    p.add_argument("--cache-dir", default=None,
                   help="cache LLM responses here so reruns are cheap")
    p.add_argument("--debug-dir", default=None,
                   help="write every raw prompt+completion here for diagnosis")
    p.add_argument("--offline", action="store_true",
                   help="never call the model; derive <topic> from the title")
    p.add_argument("--dry-run", action="store_true",
                   help="alias for --offline")
    p.add_argument("--no-summary", action="store_true",
                   help="omit the <sum> block")
    p.add_argument("--print-schema", action="store_true",
                   help="print the output tag vocabulary and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and dispatch. Returns the process exit code."""
    global VERBOSE
    args = build_parser().parse_args(argv)
    VERBOSE = args.verbose

    if args.print_schema:
        print_schema()
        return 0
    if not args.input:
        build_parser().error("input file is required (or use --print-schema)")
    if not os.path.exists(args.input):
        sys.stderr.write(f"error: no such file: {args.input}\n")
        return 2
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
