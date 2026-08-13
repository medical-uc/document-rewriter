#!/usr/bin/env python3
"""
rewrite_medical_md.py
=====================

Convert medical textbook chapters (imperfectly extracted from PDF) into tagged
markdown study documents for the knowledge-graph pipeline.

The governing rule is triple-readiness. Textbook prose is written to be read
in order by a person: sentences carry several claims at once, subjects hide
behind pronouns, and the facts held in tables and diagrams are not in
sentences at all. That is poor input for triple extraction, so the body is
rewritten rather than carried over. Each section is read whole, the
information in it is inventoried, and the section is restated as standalone
declarative sentences. Tables and mermaid diagrams are restated too, into
words. Figure and table references survive and travel with the claim they
belong to.

Nothing is added: the rewrite restates the section's information and may not
infer, generalise or supply outside knowledge. It is also not a summary, and
is expected to run longer than its source rather than shorter.

Pipeline
--------
  1. normalise  -- repair the PDF extractor's damage (see below)
  2. parse      -- scan into a flat element stream in source order
  3. metadata   -- title/objectives/summary/glossary from the source;
                   one LLM call for <topic>, which has no source counterpart
  4. rewrite    -- one call per section, per table, and per mermaid diagram
                   or data table attached to a figure
  5. split      -- one call per paragraph of a section whose rewrite has too
                   few sentences to have split its source's packed claims
  6. repair     -- one call per rewritten paragraph that still opens a
                   sentence with a back-reference, naming what it refers to
  7. audit      -- one call per section, reporting information the rewrite
                   dropped; skippable with --no-audit
  8. link       -- turn in-document 'Figure 2-1' mentions into <figref>
  9. render     -- assemble the tagged markdown, anchoring each figure and
                   table after the <con> that first references it
 10. verify     -- structural checks, any back-reference that survived
                   repair, any section still under-split, plus whatever the
                   audit found

Extraction damage repaired in stage 1
-------------------------------------
  * U+00A0 is used as the word separator throughout;
  * U+00AD stands in for a real hyphen ('Henderson<AD>Hasselbalch');
  * paragraphs are split mid-sentence wherever a page break fell.

Cost
----
Roughly 2N + T + D + S + R + 1 calls for a chapter of N sections, T tables,
D figure-attached diagrams or grids, S paragraphs needing splitting and R
needing repair, against 1 for the carry-over tool this replaced. S and R are
proportional to the damage rather than to the chapter: splitting has touched
about a quarter of sections and repair about a fifth of paragraphs. The calls
fan out over --workers, and --cache-dir makes a rerun free.

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
    python rewrite_medical_md.py chapter.md -o out.md --no-audit
    python rewrite_medical_md.py chapter.md -o out.md -v --cache-dir ./cache
    python rewrite_medical_md.py --print-schema

The backend is required: there is no offline mode, because a rewrite with no
model is not a document.

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
MAXTOK_REWRITE = 6144          # a rewrite runs longer than its source
MAXTOK_TABLE = 4096
MAXTOK_DIAGRAM = 2048
MAXTOK_SPLIT = 6144            # a split paragraph runs longer than its input
MAXTOK_REPAIR = 4096
MAXTOK_AUDIT = 2048

TEMP_TOPIC = 0.30
TEMP_CAPTION = 0.15            # conservative: stay close to the source text
TEMP_OBJECTIVES = 0.35
TEMP_SUMMARY = 0.35
# The rewrite stages restate information rather than compose, so they sample
# far more conservatively than the metadata stages.
TEMP_REWRITE = 0.20
TEMP_TABLE = 0.10
TEMP_DIAGRAM = 0.10
# Splitting redistributes claims across sentences without inventing any, so it
# samples as tightly as the restating stages.
TEMP_SPLIT = 0.10
# Repair swaps a back-reference for a name and changes nothing
# else, so it samples as tightly as the tool allows.
TEMP_REPAIR = 0.05
TEMP_AUDIT = 0.10

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
    ("topic", "meta", "One-line statement of the subject matter, inferred "
                      "from the chapter."),
    ("src", "meta", "Provenance: source filename, book, figure and table "
                    "counts, model, timestamp."),
    ("obj", "doc", "Learning objectives, from the chapter's OBJECTIVES list."),
    ("goal", "obj", "One learning objective. One per element, no bullet "
                    "marker."),
    ("sec", "doc", "One section. Attributes: id, level. Sections are flat; "
                   "depth is carried by the level attribute."),
    ("head", "sec", "Section heading, verbatim from the source."),
    ("con", "sec / tbl", "Rewritten content. Under <sec>, one claim cluster "
                         "from the section's rewrite; the grouping is the "
                         "rewriter's, not the source's paragraph breaks. "
                         "Under <tbl>, the sentences the table's grid became."),
    ("fig", "sec", "A figure. Attributes: id; label when the source numbered "
                   "it; src when it carried an image file; panel for the "
                   "'A'/'B' letter on an unnumbered inline diagram. Emitted "
                   "after the <con> that first references it."),
    ("cap", "fig / tbl", "Caption, taken from the source text."),
    ("desc", "fig", "The figure's own content in words. Holds the sentences "
                    "a mermaid diagram or an attached data table became, and "
                    "is empty for a figure that is only an image."),
    ("figref", "con", "A reference to a <fig> in this document. Attribute: "
                      "id. Wraps the mention text."),
    ("tbl", "sec", "A table. Attributes: id, label. Holds <cap> and a <con> "
                   "of prose; the source grid markup is not emitted. Emitted "
                   "after the <con> that first references it."),
    ("tblref", "con", "A reference to a <tbl> in this document. Attribute: "
                      "id. Wraps the mention text."),
    ("sum", "doc", "Chapter summary, from the source SUMMARY section."),
    ("def", "doc", "Glossary, from the source GLOSSARY section."),
    ("term", "def", "One definition, phrased as a complete sentence that "
                    "begins with the term (e.g. 'Bioethics is the area ...')."),
    ("ref", "doc", "Reference list, from the source RECOMMENDED READING."),
    ("cit", "ref", "A single citation."),
]

SCHEMA_NOTES = """
WHAT THE BODY IS

The body is rewritten, not carried over. Each section is read whole, the
information in it is inventoried, and the section is restated as prose aimed
at triple extraction:

  * one claim per sentence, shaped subject-predicate-object. Splitting a
    source's packed sentences apart should leave a section with more
    sentences than it started with, so a section that came back with too
    few is sent back to be split, and is reported by verify if it still
    reads short;
  * every sentence names its own subject, so no sentence depends on its
    neighbour to say what it is about. A sentence that still opens with a
    back-reference after the rewrite is sent back to have the name filled
    in, and is reported by verify if it survives that;
  * the source's own terms, spellings, numbers, units and conditions;
  * coordination that hides a relation is split into separate sentences;
  * transitions and rhetorical framing are dropped;
  * nothing is added, inferred or generalised.

Paragraph grouping is the rewriter's, so a <con> does not correspond to a
source paragraph. A section's <con> count is unrelated to its source's.

Tables and mermaid diagrams hold information that is not in sentences at
all, so they are restated too: a table's grid becomes a <con> of sentences
inside its <tbl>, each naming its row and its column so it stands alone, and
a mermaid diagram becomes one sentence per arrow inside the figure's <desc>.
A data table the extractor attached to a figure rather than to a <tbl> is
restated the same way, into that figure's <desc>. Neither grid markup nor
mermaid source is emitted, unless a conversion failed, in which case the
block goes out as it came and verify reports it.

COVERAGE

Because the body is rewritten, there is no verbatim text to check the output
against. Coverage is checked instead by a second pass that reads each
section's source against its rewrite and reports information the rewrite
dropped. Findings are warnings on stderr and a non-zero exit, and --no-audit
skips the pass. This is a weaker guarantee than the containment check it
replaced, and it is the price of rewriting.

MARKDOWN SYNTAX POLICY

The tags carry the document structure, so structural markdown is not
emitted:

  * <title> and <head> hold plain text, with no '#' or '##' markers.
    Heading depth is expressed by the level attribute on <sec>.
  * List items are elements (<goal>, <term>, <cit>), not '- ' bullets.

Content-level markup is preserved: inline emphasis, <sup> and <sub>, and
LaTeX spans such as $\\mathsf{pK_a}$ survive the rewrite. Nothing in element
content is entity-escaped, because escaping would corrupt that markup; only
attribute values are escaped.

CROSS-REFERENCES

A mention of a figure or table defined in this chapter is wrapped in place:

    ... clinical insights (<figref id="fig-1-1">Figure 1-1</figref>).

The rewriter is told to keep a pointer in the sentence carrying the
information the figure illustrates, and the block itself is emitted after
the <con> that first points at it. Mentions that point outside the chapter,
such as 'see Figure 40-5', are left as plain text, because there is nothing
in this document to link to.

BLOCK SHAPES

    <fig id="fig-2-1" label="Figure 2-1" src="images/a1b2c3.jpg">
    <cap>The water molecule has tetrahedral geometry.</cap>
    <desc></desc>
    </fig>

    <tbl id="tbl-2-1" label="Table 2-1">
    <cap>Bond Energies for Atoms of Biologic Significance</cap>
    <con>
    An O-O bond has a bond energy of 34 kcal/mol.
    An S-S bond has a bond energy of 51 kcal/mol.
    </con>
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
    # The extractor's machine description of the image file. It says nothing
    # about the chapter's subject matter, so it feeds caption inference only
    # and never reaches the output.
    alt_text: str = ""
    # A mermaid diagram in ``extra``, restated as sentences. Fills <desc>.
    diagram_prose: List[str] = field(default_factory=list)


@dataclass
class Table:
    """A numbered table and the source markup of its body.

    ``body`` is the extractor's <table> blob. It is the input the prose is
    built from and is not itself rendered, because a knowledge-graph
    extractor cannot read a grid.
    """

    tid: str
    label: str
    order: int
    caption: str = ""
    body: str = ""
    prose: List[str] = field(default_factory=list)


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
    """A source heading and everything that followed it.

    ``elements`` is the source in source order and is what the rewrite reads.
    ``rewritten`` is what the rewrite produced and what gets rendered: one
    string per <con>, re-chunked by claim rather than following the source's
    paragraph breaks.
    """

    sid: str
    heading: str
    level: int
    elements: List[Element] = field(default_factory=list)
    rewritten: List[str] = field(default_factory=list)

    @property
    def source_paragraphs(self) -> List[str]:
        """The section's content paragraphs, in source order."""
        return [e.text for e in self.elements if e.kind == "para"]

    @property
    def figures(self) -> List[Figure]:
        """The figures that occur in this section, in source order."""
        return [e.figure for e in self.elements
                if e.kind == "figure" and e.figure is not None]

    @property
    def tables(self) -> List[Table]:
        """The tables that occur in this section, in source order."""
        return [e.table for e in self.elements
                if e.kind == "table" and e.table is not None]


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
        """Total words across every source content paragraph."""
        return sum(word_count(p)
                   for s in self.sections for p in s.source_paragraphs)

    @property
    def rewritten_words(self) -> int:
        """Total words across every rewritten paragraph."""
        return sum(word_count(p) for s in self.sections for p in s.rewritten)


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


_JSON_ESCAPE = re.compile(r'\\(?:u[0-9a-fA-F]{4}|["\\/bfnrt])')
_JSON_CONTROL = {"\n": "\\n", "\r": "\\r", "\t": "\\t",
                 "\b": "\\b", "\f": "\\f"}


def repair_json_text(blob: str) -> str:
    """Re-escape a JSON candidate the model wrote by hand.

    The rewrite prompts ask for prose containing LaTeX, and a lone backslash
    ahead of a letter is an invalid JSON escape that rejects the whole
    object. Raw newlines inside a string reject it the same way. Both are
    escaped here; every already-valid escape is passed through untouched, so
    a well-formed blob comes back unchanged.
    """
    out: List[str] = []
    index = 0
    inside_string = False
    while index < len(blob):
        character = blob[index]
        if not inside_string:
            inside_string = character == '"'
            out.append(character)
            index += 1
            continue
        escape = _JSON_ESCAPE.match(blob, index)
        if escape:
            out.append(escape.group(0))
            index = escape.end()
            continue
        if character == "\\":
            out.append("\\\\")
        elif character == '"':
            inside_string = False
            out.append(character)
        elif ord(character) < 0x20:
            out.append(_JSON_CONTROL.get(character,
                                         "\\u%04x" % ord(character)))
        else:
            out.append(character)
        index += 1
    return "".join(out)


def extract_json(text: str) -> Any:
    """Pull the first balanced JSON object or array out of a model response.

    A candidate that fails to parse is retried once through
    repair_json_text, which fixes the escaping mistakes the model makes
    around LaTeX.
    """
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
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        pass
                    try:
                        return json.loads(repair_json_text(candidate))
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
# Stage 4: rewriting
# --------------------------------------------------------------------------

SYS_REWRITER = (
    "You restate passages from a medical textbook as declarative statements "
    "for a knowledge graph. Every sentence you write must survive being read "
    "on its own, out of order, with no neighbouring sentence for context. "
    "You answer with a single JSON object and nothing else: no preamble, no "
    "explanation, no markdown fence."
)

# Shared by both grid-reading prompts. A grid whose columns are unlabelled
# is the case that goes wrong quietly: naming only the row turns one row of
# four values into four sentences that contradict each other.
GRID_COLUMN_RULE = (
    "The columns may carry no heading of their own. Where they do not, "
    "identify each column by the value that distinguishes it, normally the "
    "one in the grid's first row, and carry that identifier into every "
    "sentence drawn from that column, as in 'In the solution whose initial "
    "pH is 5.00, ...'. Never write two sentences that assert different "
    "values for the same subject and property: a reader who sees only one "
    "of them must not be misled. If a column cannot be identified at all, "
    "leave its values out and describe only what the grid does say.\n\n"
)

# The rewrite rules, shared by the section, table and diagram prompts so the
# three stages produce prose of one shape. These are what make the output
# tractable for triple extraction: a sentence that names its subject and
# carries one relation maps onto one triple, and a sentence that leans on
# 'it' or packs three clauses does not.
REWRITE_RULES = """\
Rules:
1. One claim per sentence. Each sentence states a single relationship
   between a named subject and a named object.
2. Name the subject explicitly in every sentence. A sentence that needs the
   previous sentence to identify its subject is wrong. This rules out bare
   pronouns ('it', 'they', 'them'), and it equally rules out a demonstrative
   in front of a noun ('this condition', 'these groups', 'such
   interactions'), because the noun still does not say which one.
     Wrong: Nephrogenic diabetes insipidus resists vasopressin. This
            condition prevents the kidneys from concentrating urine.
     Right: Nephrogenic diabetes insipidus resists vasopressin.
            Nephrogenic diabetes insipidus prevents the kidneys from
            concentrating urine.
   Repeating a long name in consecutive sentences is correct here, and is
   always preferred to a pronoun.
3. Use the source's own terminology, spelled exactly as the source spells
   it. Never substitute a synonym for a technical term. Give an
   abbreviation's full form the first time it appears, as 'antidiuretic
   hormone (ADH)', and use it consistently after that.
4. Reproduce every number, unit, range, percentage and condition exactly.
   Keep the condition attached to the claim it governs, as in 'At 25 degrees
   Celsius, the dielectric constant of water is 78.5.'
5. Split coordination that hides a relation. 'A and B cause C' becomes two
   sentences. 'A, which does X, causes C' becomes two sentences.
6. Prefer the active voice and put the entity first.
7. Drop transitions, rhetorical framing and asides that carry no fact
   ('It follows that', 'As we shall see', 'Interestingly').
8. Add nothing. Every statement must be supported by the passage given to
   you. Do not infer, generalise, or supply outside knowledge.
9. Keep inline markup as the source has it: <sup>, <sub>, and LaTeX spans
   such as $\\mathsf{pK_a}$."""


def _figure_inventory(section: Section) -> str:
    """List the section's referenceable figures and tables for a prompt.

    Only labelled ones appear: an unnumbered inline diagram has no mention
    string the prose could use, so offering it would invite an invented one.
    """
    lines: List[str] = []
    for fig in section.figures:
        if fig.label:
            lines.append(f"  {fig.label}: {fig.caption or '(no caption)'}")
    for tbl in section.tables:
        if tbl.label:
            lines.append(f"  {tbl.label}: {tbl.caption or '(no caption)'}")
    return "\n".join(lines)


def prompt_section(section: Section) -> str:
    """Build the prompt that rewrites one section into knowledge-graph prose."""
    body = "\n\n".join(section.source_paragraphs)
    inventory = _figure_inventory(section)

    parts = [
        "Below is one section of a medical textbook chapter, under the "
        "heading shown. Read all of it and take note of every piece of "
        "information it contains: entities, properties, values, causes, "
        "effects, mechanisms, classifications and conditions.\n",
        f"HEADING: {section.heading}\n",
        f"SECTION TEXT:\n{body}\n",
    ]

    if inventory:
        parts.append(
            "This section contains the following figures and tables:\n"
            f"{inventory}\n\n"
            "Where the passage points the reader at one of them, keep that "
            "pointer in the sentence carrying the information it illustrates, "
            "written exactly as the label above and in parentheses, for "
            "example '(Figure 2-1)'. Write the label as plain text; never "
            "write a tag. Do not refer to a figure or table that is not "
            "listed above, except where the passage itself points outside "
            "this chapter, which you should carry over unchanged.\n"
        )

    parts.append(
        "Rewrite the section using the information you noted. Cover every "
        "piece of it. Do not summarise and do not shorten: the rewrite "
        "exists to make the information easier to read as statements, not "
        "to make it briefer.\n"
    )
    parts.append(REWRITE_RULES)
    parts.append(
        "\nGroup the sentences into paragraphs, one paragraph per subject or "
        "closely related group of subjects. A paragraph here is a grouping "
        "of standalone sentences, not connected prose: the sentences in it "
        "carry no transitions, no back-references and no reading order, and "
        "every one of them still names its own subject. Assume each "
        "sentence will be pulled out of its paragraph and read by itself. "
        "Answer with JSON of the form "
        '{"paragraphs": ["...", "..."]}.'
    )
    return "\n".join(parts)


def _parse_paragraphs(raw: str, key: str, tag: str) -> List[str]:
    """Read a list of strings out of a model response.

    Falls back to splitting the raw completion on blank lines when the JSON
    is unparseable, because a model that ignored the output contract has
    usually still written usable prose. A response that did attempt JSON and
    still will not parse returns nothing instead, so the caller fails loudly
    rather than write braces into the document as prose.
    """
    try:
        data = extract_json(raw)
    except ValueError:
        body = raw.strip()
        if body.startswith("{") or body.startswith("["):
            log(f"{tag}: JSON response is unparseable even after repair")
            return []
        log(f"{tag}: no JSON in response; falling back to blank-line split")
        chunks = re.split(r"\n\s*\n", body)
        return [c.strip() for c in chunks if c.strip()]

    if isinstance(data, dict):
        data = data.get(key, [])
    if isinstance(data, str):
        data = [data]
    if not isinstance(data, list):
        return []
    return [str(item).strip() for item in data if str(item).strip()]


def rewrite_sections(client: LLMClient, doc: SourceDoc, workers: int) -> None:
    """Rewrite every section's prose in place.

    Raises LLMError if a section with source paragraphs comes back empty,
    rather than emit a section whose content silently vanished.
    """
    todo = [s for s in doc.sections if s.source_paragraphs]
    if not todo:
        return

    failures: List[str] = []

    def one(section: Section) -> None:
        """Rewrite a single section, recording a failure instead of raising."""
        raw = client.chat(SYS_REWRITER, prompt_section(section),
                          max_tokens=MAXTOK_REWRITE, temperature=TEMP_REWRITE,
                          tag=f"rewrite:{section.sid}")
        section.rewritten = _parse_paragraphs(raw, "paragraphs",
                                              f"rewrite:{section.sid}")
        if not section.rewritten:
            failures.append(section.sid)

    _run_parallel(one, todo, workers, "rewrite")

    if failures:
        raise LLMError("no usable rewrite for section(s): "
                       + ", ".join(failures))


def prompt_table(tbl: Table, doc: SourceDoc) -> str:
    """Build the prompt that turns one table into standalone sentences."""
    return (
        f"Below is a table from a chapter of a medical textbook titled "
        f"'{doc.title}'. A knowledge-graph extractor cannot read a grid, so "
        "the table has to become sentences.\n\n"
        f"LABEL: {tbl.label}\n"
        f"CAPTION: {tbl.caption or '(none)'}\n\n"
        f"TABLE:\n{tbl.body}\n\n"
        "Write one sentence for every value in the table. Each sentence "
        "must name the row it came from and the quantity or property the "
        "column represents, so that it stands alone without the grid. For "
        "a table of bond energies, that reads 'An O-O bond has a bond "
        "energy of 34 kcal/mol.' Work through the rows in order and leave "
        "no value out. Where a column heading names a unit, put the unit in "
        "the sentence.\n\n"
        + GRID_COLUMN_RULE
        + REWRITE_RULES +
        '\n\nAnswer with JSON of the form {"sentences": ["...", "..."]}.'
    )


def rewrite_tables(client: LLMClient, doc: SourceDoc, workers: int) -> None:
    """Turn every table body into prose, in place.

    A table whose prose fails is logged and left with empty prose; verify
    reports it. One unreadable grid should not lose the chapter.
    """
    todo = [t for t in doc.tables if t.body]
    if not todo:
        return

    def one(tbl: Table) -> None:
        """Convert a single table, logging and skipping on backend failure."""
        try:
            raw = client.chat(SYS_REWRITER, prompt_table(tbl, doc),
                              max_tokens=MAXTOK_TABLE, temperature=TEMP_TABLE,
                              tag=f"table:{tbl.tid}")
        except LLMError as exc:                           # noqa: BLE001
            log(f"table prose failed for {tbl.tid}: {exc}")
            return
        tbl.prose = _parse_paragraphs(raw, "sentences", f"table:{tbl.tid}")

    _run_parallel(one, todo, workers, "tables")


# A diagram carried inside a <details> block. The extractor emits these as
# mermaid, which is the one thing in a chapter that genuinely describes its
# own image, so its prose fills <desc>.
MERMAID_RE = re.compile(r"```\s*mermaid\b(.*?)```", re.S | re.I)
EXTRA_TABLE_RE = re.compile(r"<table\b", re.I)


def mermaid_source(extra: str) -> str:
    """Return the mermaid body inside a figure's extra block, or ''."""
    m = MERMAID_RE.search(extra or "")
    return m.group(1).strip() if m else ""


def figure_extra_kind(extra: str) -> str:
    """Classify a figure's extra block as 'mermaid', 'table' or ''.

    The extractor attaches whatever structured block trailed the figure.
    Both kinds carry information a triple extractor cannot read, so both
    become <desc> prose; '' means there is nothing to convert.
    """
    if mermaid_source(extra):
        return "mermaid"
    if extra and EXTRA_TABLE_RE.search(extra):
        return "table"
    return ""


def prompt_diagram(fig: Figure, doc: SourceDoc) -> str:
    """Build the prompt that turns one figure's extra block into sentences.

    Dispatches on figure_extra_kind, because a mermaid graph is read arrow
    by arrow and a grid is read cell by cell.
    """
    head = (
        f"LABEL: {fig.label or '(unnumbered)'}\n"
        f"CAPTION: {fig.caption or '(none)'}\n\n"
    )
    if figure_extra_kind(fig.extra) == "table":
        task = (
            f"Below is a data table attached to a figure in a chapter of a "
            f"medical textbook titled '{doc.title}'. A knowledge-graph "
            "extractor cannot read a grid, so the table has to become "
            "sentences.\n\n"
            f"{head}"
            f"TABLE:\n{fig.extra}\n\n"
            "Write one sentence for every value in the grid. Name the "
            "value's row and its column in the sentence, so the sentence "
            "stands on its own without the grid. Carry every number and "
            "unit across exactly. Cover every cell that holds a value.\n\n"
            + GRID_COLUMN_RULE
        )
    else:
        task = (
            f"Below is a mermaid diagram from a chapter of a medical "
            f"textbook titled '{doc.title}'. A knowledge-graph extractor "
            "cannot read mermaid, so the diagram has to become "
            "sentences.\n\n"
            f"{head}"
            f"DIAGRAM:\n{mermaid_source(fig.extra)}\n\n"
            "Write one sentence for every arrow in the diagram, using the "
            "node labels exactly as the diagram spells them, and naming the "
            "relationship the arrow asserts in the language of the caption. "
            "Cover every arrow. Then write one closing sentence naming what "
            "the diagram as a whole relates.\n\n"
        )
    return (
        task +
        "Do not describe the figure as a figure: write the information "
        "itself, not 'the figure shows'.\n\n"
        + REWRITE_RULES +
        '\n\nAnswer with JSON of the form {"sentences": ["...", "..."]}.'
    )


def rewrite_diagrams(client: LLMClient, doc: SourceDoc, workers: int) -> None:
    """Turn every figure's structured extra block into prose, in place.

    Covers both mermaid diagrams and the data tables the extractor attaches
    to a figure. A figure whose conversion fails keeps its extra block and
    is reported by verify.
    """
    todo = [f for f in doc.figures if figure_extra_kind(f.extra)]
    if not todo:
        return

    def one(fig: Figure) -> None:
        """Convert one extra block, logging and skipping on backend failure."""
        tag = f"{figure_extra_kind(fig.extra)}:{fig.fid}"
        try:
            raw = client.chat(SYS_REWRITER, prompt_diagram(fig, doc),
                              max_tokens=MAXTOK_DIAGRAM,
                              temperature=TEMP_DIAGRAM,
                              tag=tag)
        except LLMError as exc:                           # noqa: BLE001
            log(f"figure prose failed for {fig.fid}: {exc}")
            return
        fig.diagram_prose = _parse_paragraphs(raw, "sentences", tag)

    _run_parallel(one, todo, workers, "figure extras")


# --------------------------------------------------------------------------
# Stage 5: claim splitting
# --------------------------------------------------------------------------

# A rewrite that obeys the one-claim-per-sentence rule ends up with more
# sentences than its source, because textbook prose packs several claims into
# one sentence. A section that came back with barely more sentences than it
# started with therefore did not split anything, whatever its wording suggests.
# A section at or below this ratio is reported as under-split.
SPLIT_RATIO_FLOOR = 1.2

# Sending a section for splitting is cheap and refusable, so the stage uses a
# more generous threshold than the report does. Splitting is worth attempting
# well before a section is bad enough to be worth complaining about, and a
# section between the two thresholds gains sentences without ever having been
# reported. A result that comes back no better is refused by the guards and the
# original kept, so the wider net costs calls rather than quality.
SPLIT_TRIGGER_RATIO = 1.5

# Below this many source claims the ratio quantises too coarsely to mean
# anything. A six-claim section fails at seven sentences and passes at eight,
# and that gap is not a real distinction about how well it was split.
SPLIT_MIN_SOURCE_SENTENCES = 10

# The converter writes a displayed equation out as an alt-text line of its own,
# in a handful of shapes: 'An equation reads, ...', 'A reaction reads, ...',
# 'Two reversible reactions of ...', 'Two calculations to find ...'.
EQUATION_ALT_TEXT_RE = re.compile(
    r"^(?:An?|Two|Three|Four) (?:\w+ )?"
    r"(?:equations?|reactions?|calculations?)\b")

# A split redistributes claims; it does not invent them. Sentence count is the
# wrong thing to bound here, because multiplying sentences is the whole point:
# one packed sentence legitimately becomes five. Words are the honest measure,
# since splitting only repeats subjects while padding introduces new content.
# Observed splits run to 2.6x words, so this refuses well clear of them.
SPLIT_WORD_CEILING = 3.0

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")

# LaTeX spans and the sup/sub tags carry chemical formulae, so a split that
# quietly unwraps '$\mathsf{H}_3\mathsf{O}^+$' into bare text has corrupted
# the claim even though every word survived.
MARKUP_RE = re.compile(r"\$[^$]+\$|</?su[bp]>")

# Told to copy formulae verbatim, the model still unwraps them, doubles the
# backslash in '\times', and drops a closing delimiter so the span swallows the
# sentences after it. Masking each formula behind an opaque token takes away
# the opportunity instead of arguing about it. A whole '<sup>2</sup>' is masked
# as one unit so its digit cannot be stranded from its tags.
MASKABLE_RE = re.compile(r"\$[^$]+\$|<(su[bp])>.*?</\1>|</?su[bp]>")

PLACEHOLDER_TEMPLATE = "[FORMULA{}]"

PLACEHOLDER_RE = re.compile(r"\[FORMULA(\d+)\]")


def conversion_artefact(sentence: str) -> bool:
    """Report whether a counted source sentence is PDF-conversion debris.

    An odd number of '$' means the sentence splitter cut through a formula,
    because the converter writes math with spaces around its punctuation, as
    in '$1 . 8 \\times 10^{-9}$'. A trailing colon marks the clause that
    introduces a displayed equation, and the equation arrives as an alt-text
    line of its own. None of the three is a claim.
    """
    if sentence.count("$") % 2 == 1:
        return True
    if sentence.rstrip().endswith(":"):
        return True
    return bool(EQUATION_ALT_TEXT_RE.match(sentence))


def source_sentence_count(section: Section) -> int:
    """Count the claims in a section's source, ignoring what is not a claim.

    Fragments of three words or fewer are page furniture left by the PDF
    conversion, and `conversion_artefact` catches the rest of that debris.
    Counting any of it would understate the split ratio, and it concentrates
    in exactly the formula-dense sections the ratio judges most harshly.
    """
    return sum(1 for paragraph in section.source_paragraphs
               for sentence in split_sentences(paragraph)
               if len(sentence.split()) > 3
               and not conversion_artefact(sentence))


def rewritten_sentence_count(section: Section) -> int:
    """Count the sentences across a section's rewritten paragraphs."""
    return sum(len(split_sentences(p)) for p in section.rewritten)


def split_ratio(section: Section) -> Optional[float]:
    """Return rewritten sentences per source claim, or None if unmeasurable.

    None means the section offers no usable baseline, either because it holds
    no countable source claims or because it holds too few for the ratio to
    discriminate. Such a section is left alone rather than judged.
    """
    source_count = source_sentence_count(section)
    if source_count < SPLIT_MIN_SOURCE_SENTENCES:
        return None
    return rewritten_sentence_count(section) / source_count


def understretched_sections(doc: SourceDoc, floor: float) -> List[Section]:
    """Return the sections whose split ratio falls below `floor`.

    `floor` is the caller's, because the split stage casts a wider net than
    the report does. A section whose ratio is unmeasurable is never returned.
    """
    out: List[Section] = []
    for section in doc.sections:
        if not section.rewritten:
            continue
        ratio = split_ratio(section)
        if ratio is not None and ratio < floor:
            out.append(section)
    return out


def numbers_in(text: str) -> List[str]:
    """Return every numeric literal in a string, in order of appearance."""
    return NUMBER_RE.findall(text)


def normalise_markup(token: str) -> str:
    """Reduce a markup token to the form worth comparing across a split.

    Whitespace inside math mode does not render, and the source's spacing is
    erratic, so it is dropped. Punctuation stranded inside the delimiters is
    dropped too: moving a trailing comma out of '$pK_a ,$' fixes the span
    rather than damaging it. What survives is the formula itself, so a real
    corruption such as '\\times' becoming '\\\\times' still compares unequal.
    """
    if not token.startswith("$"):
        return token
    inner = "".join(token[1:-1].split())
    return "$" + inner.strip(",.;:") + "$"


def markup_in(text: str) -> List[str]:
    """Return every LaTeX span and sup/sub tag in a string, normalised."""
    return [normalise_markup(t) for t in MARKUP_RE.findall(text)]


def mask_markup(text: str) -> Tuple[str, List[str]]:
    """Replace every formula and sup/sub element with an opaque placeholder.

    Returns the masked text and the spans taken out of it, ordered so that a
    span's position in the list matches the number in its placeholder.
    `unmask_markup` reverses this.
    """
    spans: List[str] = []

    def take(match: "re.Match[str]") -> str:
        """Record one span and yield the placeholder standing in for it."""
        spans.append(match.group(0))
        return PLACEHOLDER_TEMPLATE.format(len(spans))

    return MASKABLE_RE.sub(take, text), spans


def unmask_markup(text: str, spans: List[str]) -> str:
    """Put the masked spans back where their placeholders now sit.

    A placeholder numbered past the end of `spans` is left as it stands, so a
    number the model invented surfaces as visibly unrestored text rather than
    as a formula silently attached to the wrong claim.
    """
    def put(match: "re.Match[str]") -> str:
        """Return the span a single placeholder stands for."""
        index = int(match.group(1))
        if 1 <= index <= len(spans):
            return spans[index - 1]
        return match.group(0)

    return PLACEHOLDER_RE.sub(put, text)


def placeholder_leaked(original: str, candidate: str) -> bool:
    """Report whether a split result carries a placeholder that never restored.

    A placeholder the model renumbered or pulled apart, such as '[FORMULA 1]',
    does not match on the way back, and the bare word left behind is the
    visible trace of it. The original is checked too, so a section that talks
    about formulae in its own words is not mistaken for a damaged mask.
    """
    return "FORMULA" in candidate and "FORMULA" not in original


def prompt_split(section: Section, paragraph: str) -> str:
    """Build the prompt that splits one paragraph's packed sentences apart.

    The heading goes in as context so the model can name a subject that the
    paragraph itself refers to only by role. `paragraph` is expected to have
    been through `mask_markup` already, since the prompt tells the model what
    the placeholders mean.
    """
    return (
        "Below is a paragraph of statements from a medical textbook, already "
        "rewritten once for a knowledge graph. Some of its sentences still "
        "carry more than one claim. A knowledge graph stores one relation per "
        "statement, so a sentence asserting two things has to become two "
        "sentences.\n\n"
        f"HEADING: {section.heading}\n\n"
        f"PARAGRAPH:\n{paragraph}\n\n"
        "Split every sentence that makes more than one assertion into one "
        "sentence per assertion. 'Collagen contains glycine at every third "
        "residue, and this spacing lets the three chains pack tightly.' "
        "becomes 'Collagen contains glycine at every third residue.' and "
        "'The glycine spacing in collagen lets the three chains pack "
        "tightly.'\n\n"
        "Give every sentence you produce its own subject, named in full. Do "
        "not start a sentence with 'this', 'these', 'it' or 'they'; repeat "
        "the name instead, however long it is.\n\n"
        "A list of items sharing one predicate is a single claim and stays "
        "one sentence: 'Vitamin C deficiency causes bleeding gums, swelling "
        "joints, and poor wound healing.' is already correct. Split on "
        "assertions, not on commas.\n\n"
        "Add nothing. Every claim in your answer must be in the paragraph "
        "above, and every claim in the paragraph must be in your answer. "
        "Keep each number, unit, range, condition, and each figure or table "
        "mention such as '(Figure 2-1)' exactly as written, attached to the "
        "claim it belongs to. Leave a sentence that already makes exactly "
        "one claim exactly as it is, character for character.\n\n"
        "Placeholders such as [FORMULA1] stand for chemical formulae that "
        "have been taken out of the paragraph. Copy each one across exactly "
        "as written, its number included, attached to the claim it belongs "
        "to. Do not write out what a placeholder might stand for, do not "
        "renumber one, and do not put spaces inside its brackets. A "
        "placeholder may be the subject of a sentence, and it counts as a "
        "term, so a claim about [FORMULA1] names [FORMULA1].\n\n"
        "Answer with JSON of the form {\"sentences\": [\"...\", \"...\"]}, "
        "holding the whole paragraph in its original order."
    )


def split_claims(client: LLMClient, doc: SourceDoc, workers: int) -> None:
    """Split multi-claim sentences apart, in place, across weak sections.

    Only paragraphs in sections below `SPLIT_TRIGGER_RATIO` are sent, so the
    cost tracks the number of weakly split sections rather than the chapter.
    That threshold is looser than the one `verify` reports against, so a
    section is attempted before it is bad enough to be worth reporting.
    Formulae are masked behind placeholders for the call and restored after it.
    A paragraph is left as it was when the call fails, when the result loses
    sentences, numbers or formula markup, when a placeholder fails to restore,
    or when its word count expands past the padding ceiling.
    """
    todo: List[Tuple[Section, int]] = []
    for section in understretched_sections(doc, SPLIT_TRIGGER_RATIO):
        for index in range(len(section.rewritten)):
            todo.append((section, index))
    if not todo:
        return

    def one(item: Tuple[Section, int]) -> None:
        """Split one paragraph, keeping the original on any failure."""
        section, index = item
        original = section.rewritten[index]
        masked, spans = mask_markup(original)
        tag = f"split:{section.sid}#{index}"
        try:
            raw = client.chat(SYS_REWRITER, prompt_split(section, masked),
                              max_tokens=MAXTOK_SPLIT,
                              temperature=TEMP_SPLIT,
                              tag=tag)
        except LLMError as exc:                           # noqa: BLE001
            log(f"split failed for {tag}: {exc}")
            return
        sentences = _parse_paragraphs(raw, "sentences", tag)
        if not sentences:
            return
        before = len(split_sentences(original))
        if len(sentences) < before:
            log(f"{tag}: split returned fewer sentences; keeping original")
            return
        candidate = unmask_markup(" ".join(sentences), spans)
        if placeholder_leaked(original, candidate):
            log(f"{tag}: split damaged a formula placeholder; "
                "keeping original")
            return
        original_words = len(original.split())
        if len(candidate.split()) > original_words * SPLIT_WORD_CEILING:
            log(f"{tag}: split expanded {original_words} words to "
                f"{len(candidate.split())}; keeping original")
            return
        # Splitting moves numbers between sentences but must never lose one, so
        # a dropped literal is the cheapest reliable sign the claims drifted.
        missing = _missing_numbers(original, candidate)
        if missing:
            log(f"{tag}: split dropped {', '.join(missing)}; keeping original")
            return
        unwrapped = _missing_markup(original, candidate)
        if unwrapped:
            extra = (f" and {len(unwrapped) - 1} more"
                     if len(unwrapped) > 1 else "")
            log(f"{tag}: split dropped markup {unwrapped[0]}{extra}; "
                "keeping original")
            return
        section.rewritten[index] = candidate

    _run_parallel(one, todo, workers, "claim splitting")


def _missing_tokens(original: str, candidate: str, extract) -> List[str]:
    """Return the tokens `extract` finds in `original` that `candidate` lost.

    Counts occurrences, so a token dropped from one of two sentences is still
    reported even when another sentence still carries it.
    """
    remaining = list(extract(candidate))
    missing: List[str] = []
    for token in extract(original):
        if token in remaining:
            remaining.remove(token)
        else:
            missing.append(token)
    return missing


def _missing_numbers(original: str, candidate: str) -> List[str]:
    """Return the numeric literals of `original` that `candidate` lost."""
    return _missing_tokens(original, candidate, numbers_in)


def _missing_markup(original: str, candidate: str) -> List[str]:
    """Return the LaTeX spans and sup/sub tags that `candidate` lost."""
    return _missing_tokens(original, candidate, markup_in)


# --------------------------------------------------------------------------
# Stage 6: back-reference repair
# --------------------------------------------------------------------------

# A sentence opening with one of these needs its neighbour to say what it is
# about, which is exactly what a triple extractor cannot do. The rewrite
# rules forbid them and the model writes them anyway, so they are found here
# by inspection and sent back one paragraph at a time.
BACKREF_RE = re.compile(
    r"^(?:This|That|These|Those|Such|They|Them|Their|Theirs|It|Its"
    r"|He|Him|His|She|Her|Hers)\b")


def split_sentences(text: str) -> List[str]:
    """Split a paragraph into sentences on terminal punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def backreferences(paragraph: str) -> List[str]:
    """Return the paragraph's sentences that open with a back-reference."""
    return [s for s in split_sentences(paragraph) if BACKREF_RE.match(s)]


def prompt_repair(section: Section, paragraph: str,
                  offenders: List[str]) -> str:
    """Build the prompt that names the subject of one paragraph's back-
    references.

    The whole section rewrite goes in as context, because the name a
    back-reference stands for often sits in an earlier paragraph.
    """
    listed = "\n".join(f"  - {s}" for s in offenders)
    context = "\n\n".join(section.rewritten)
    return (
        "Below is a passage of standalone statements from a medical "
        "textbook, and one paragraph taken from it. Some sentences in that "
        "paragraph begin with a back-reference, meaning a word that only "
        "says which thing it is about if you have read the sentence "
        "before it. Every sentence has to stand on its own.\n\n"
        f"HEADING: {section.heading}\n\n"
        f"WHOLE PASSAGE, for working out what each back-reference names:\n"
        f"{context}\n\n"
        f"PARAGRAPH TO REPAIR:\n{paragraph}\n\n"
        "SENTENCES THAT BEGIN WITH A BACK-REFERENCE:\n"
        f"{listed}\n\n"
        "Rewrite only the sentences listed above. In each one, replace the "
        "opening back-reference with the name of the thing it refers to, "
        "taken from the passage. 'This condition prevents the kidneys from "
        "concentrating urine.' becomes 'Nephrogenic diabetes insipidus "
        "prevents the kidneys from concentrating urine.' if that is what "
        "the passage says the condition is.\n\n"
        "Leave every other sentence exactly as it is, character for "
        "character. Do not add a claim, drop a claim, merge sentences or "
        "split them. Keep every number, unit, piece of markup, and every "
        "figure or table mention such as '(Figure 2-1)' exactly as "
        "written. If the passage does not say what a back-reference names, "
        "leave that sentence unchanged rather than guess.\n\n"
        "Answer with JSON of the form {\"sentences\": [\"...\", \"...\"]}, "
        "holding every sentence of the paragraph in its original order, "
        "repaired ones and untouched ones alike."
    )


def repair_backreferences(client: LLMClient, doc: SourceDoc,
                          workers: int) -> None:
    """Replace sentence-initial back-references with the names they stand
    for, in place.

    Only paragraphs that fail the BACKREF_RE check are sent, so the cost is
    proportional to the damage rather than to the chapter. A paragraph whose
    repair fails, comes back empty, or comes back having lost sentences is
    left as it was.
    """
    todo: List[Tuple[Section, int, List[str]]] = []
    for section in doc.sections:
        for index, paragraph in enumerate(section.rewritten):
            offenders = backreferences(paragraph)
            if offenders:
                todo.append((section, index, offenders))
    if not todo:
        return

    def one(item: Tuple[Section, int, List[str]]) -> None:
        """Repair one paragraph, keeping the original on any failure."""
        section, index, offenders = item
        original = section.rewritten[index]
        tag = f"repair:{section.sid}#{index}"
        try:
            raw = client.chat(SYS_REWRITER,
                              prompt_repair(section, original, offenders),
                              max_tokens=MAXTOK_REPAIR,
                              temperature=TEMP_REPAIR,
                              tag=tag)
        except LLMError as exc:                           # noqa: BLE001
            log(f"repair failed for {tag}: {exc}")
            return
        sentences = _parse_paragraphs(raw, "sentences", tag)
        # A short return means the model dropped claims instead of naming
        # subjects, which loses more than the back-references cost.
        if len(sentences) < len(split_sentences(original)):
            log(f"{tag}: repair returned fewer sentences; keeping original")
            return
        section.rewritten[index] = " ".join(sentences)

    _run_parallel(one, todo, workers, "back-reference repair")


# --------------------------------------------------------------------------
# Stage 7: coverage audit
# --------------------------------------------------------------------------

SYS_AUDITOR = (
    "You check a rewritten passage against its source for lost information. "
    "You are strict and literal: you report only information the source "
    "states and the rewrite does not. You answer with a single JSON object "
    "and nothing else."
)


def prompt_audit(section: Section) -> str:
    """Build the prompt comparing one section's rewrite against its source."""
    return (
        "SOURCE PASSAGE:\n" + "\n\n".join(section.source_paragraphs) + "\n\n"
        "REWRITE:\n" + "\n\n".join(section.rewritten) + "\n\n"
        "The rewrite was supposed to carry every piece of information in the "
        "source passage. List the pieces it dropped. A piece counts as "
        "dropped only if the source states it and the rewrite states neither "
        "it nor an equivalent. Wording is allowed to differ; sentence order "
        "and paragraph grouping are allowed to differ; a fact split across "
        "several rewritten sentences is not dropped. Ignore transitions and "
        "rhetorical framing, which the rewrite was told to remove.\n\n"
        "Report each dropped piece as one short sentence naming the missing "
        "information. If nothing was dropped, answer with an empty list. "
        'Answer with JSON of the form {"missing": ["...", "..."]}.'
    )


def audit_sections(client: LLMClient, doc: SourceDoc,
                   workers: int) -> Dict[str, List[str]]:
    """Report, per section id, the information the rewrite dropped.

    A section whose audit call fails is logged and reported as unaudited
    rather than passed, so a backend wobble cannot read as a clean bill.
    """
    todo = [s for s in doc.sections if s.source_paragraphs and s.rewritten]
    findings: Dict[str, List[str]] = {}
    if not todo:
        return findings

    def one(section: Section) -> None:
        """Audit a single section."""
        try:
            raw = client.chat(SYS_AUDITOR, prompt_audit(section),
                              max_tokens=MAXTOK_AUDIT, temperature=TEMP_AUDIT,
                              tag=f"audit:{section.sid}")
        except LLMError as exc:                           # noqa: BLE001
            findings[section.sid] = [f"audit did not run: {exc}"]
            return
        missing = _parse_paragraphs(raw, "missing", f"audit:{section.sid}")
        if missing:
            findings[section.sid] = missing

    _run_parallel(one, todo, workers, "audit")
    return findings


# --------------------------------------------------------------------------
# Stage 8: cross-reference linking
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
# Stage 9: rendering
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

    if fig.diagram_prose:
        out.append("<desc>")
        out.extend(link_references(s, known) for s in fig.diagram_prose)
        out.append("</desc>")
    else:
        out.append("<desc></desc>")

    # A converted extra now lives in <desc>. One that failed to convert goes
    # out as it came, so a backend failure loses nothing but the shaping.
    if fig.extra and not fig.diagram_prose:
        out.append("")
        out.append(fig.extra)
    out.append("</fig>")
    return "\n".join(out)


def render_table(tbl: Table, known: Dict[str, str]) -> str:
    """Render one <tbl>. ``known`` links cross-references inside the caption.

    The body is the prose the grid became. The source <table> markup is not
    emitted, because the whole point of the conversion is that a triple
    extractor cannot read it.
    """
    attrs = f'id="{escape_attr(tbl.tid)}"'
    if tbl.label:
        attrs += f' label="{escape_attr(tbl.label)}"'
    out = [f"<tbl {attrs}>"]
    out.append(f"<cap>{_caption(tbl.caption, known)}</cap>")
    if tbl.prose:
        out.append("<con>")
        out.extend(link_references(s, known) for s in tbl.prose)
        out.append("</con>")
    out.append("</tbl>")
    return "\n".join(out)


def anchor_blocks(section: Section,
                  paragraphs: Sequence[str]) -> Dict[int, List[Element]]:
    """Decide which rewritten paragraph each figure and table follows.

    Returns a map from paragraph index to the elements emitted after it. Key
    -1 is the trailing bucket, used when a section has no paragraphs to
    anchor to.

    A block the prose references is placed after the paragraph that first
    references it, which is what keeps a figure next to the claim it
    illustrates once the rewrite has re-chunked the section. A block nothing
    references -- an unnumbered inline diagram has no mention string, so this
    is its normal case -- keeps its source position proportionally, so it
    still lands among the paragraphs it sat between.
    """
    anchors: Dict[int, List[Element]] = {}
    if not paragraphs:
        anchors[-1] = [e for e in section.elements if e.kind != "para"]
        return anchors

    seen_paras = 0
    total_paras = len(section.source_paragraphs)
    for element in section.elements:
        if element.kind == "para":
            seen_paras += 1
            continue

        if element.kind == "figure" and element.figure is not None:
            ref_id = element.figure.fid
        elif element.kind == "table" and element.table is not None:
            ref_id = element.table.tid
        else:
            ref_id = ""

        target: Optional[int] = None
        if ref_id:
            needle = f'id="{escape_attr(ref_id)}"'
            target = next((i for i, p in enumerate(paragraphs) if needle in p),
                          None)
        if target is None:
            # Same fraction of the way through the rewrite as through the
            # source, minus one because the block follows that paragraph.
            share = seen_paras / total_paras if total_paras else 1.0
            target = min(len(paragraphs) - 1,
                         max(0, int(round(share * len(paragraphs))) - 1))
        anchors.setdefault(target, []).append(element)
    return anchors


def render_section(section: Section, known: Dict[str, str]) -> str:
    """Render one <sec> from its rewritten paragraphs, with blocks anchored."""
    out = [f'<sec id="{escape_attr(section.sid)}" level="{section.level}">',
           f"<head>{clean_heading(section.heading)}</head>", ""]

    paragraphs = [link_references(p, known) for p in section.rewritten]
    anchors = anchor_blocks(section, paragraphs)

    def emit(element: Element) -> None:
        """Append one figure or table block."""
        if element.kind == "figure" and element.figure is not None:
            out.append(render_figure(element.figure, known))
        elif element.kind == "table" and element.table is not None:
            out.append(render_table(element.table, known))
        out.append("")

    for index, paragraph in enumerate(paragraphs):
        out.append("<con>")
        out.append(paragraph)
        out.append("</con>")
        out.append("")
        for element in anchors.get(index, ()):
            emit(element)

    for element in anchors.get(-1, ()):
        emit(element)

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
    "figure-alt",     # describes the image file, not the chapter's subject
    "figure-head", "figure-image",
    "figure-detail",  # a mermaid diagram here reaches <desc> as prose
    "table-head",
    "table-body",     # reaches <tbl> as prose, not as markup
}

EMITTED_DISPOSITIONS = {
    "content", "caption", "objectives", "summary", "glossary", "references",
}


def verify(output: str, doc: SourceDoc,
           audit: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """Check the document's structure and report what the audit found.

    Content preservation is no longer checkable here. The body is rewritten,
    so a source paragraph has no verbatim counterpart to look for; ``audit``
    carries the per-section coverage findings that replaced that check, and
    is None when --no-audit skipped the pass.
    """
    problems: List[str] = []

    known_dispositions = SILENT_DISPOSITIONS | EMITTED_DISPOSITIONS
    for block in doc.blocks:
        if block.disposition not in known_dispositions:
            problems.append(f"UNACCOUNTED source block ({block.disposition}): "
                            f"{block.text[:100]!r}")

    for section in doc.sections:
        if section.source_paragraphs and not section.rewritten:
            problems.append(f"section {section.sid} has "
                            f"{len(section.source_paragraphs)} source "
                            "paragraph(s) but no rewrite")

    for tbl in doc.tables:
        if tbl.body and not tbl.prose:
            problems.append(f"table {tbl.tid} was not converted to prose")
    for fig in doc.figures:
        kind = figure_extra_kind(fig.extra)
        if kind and not fig.diagram_prose:
            problems.append(f"{kind} attached to {fig.fid} was not "
                            "converted to prose")

    for section in doc.sections:
        for paragraph in section.rewritten:
            for sentence in backreferences(paragraph):
                problems.append(f"BACK-REFERENCE ({section.sid}): "
                                f"{sentence}")

    for section in understretched_sections(doc, SPLIT_RATIO_FLOOR):
        problems.append(
            f"UNSPLIT ({section.sid}): {source_sentence_count(section)} "
            f"source sentences became {rewritten_sentence_count(section)}, "
            f"a ratio of {split_ratio(section):.2f}")

    for section_id, missing in sorted((audit or {}).items()):
        for item in missing:
            problems.append(f"INFORMATION LOST ({section_id}): {item}")

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

    for tag in ("doc", "meta", "obj", "sum", "def", "ref", "con", "desc"):
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
    is unreachable or the rewrite came back unusable, 2 if the input yielded
    no content, 1 if verification raised warnings, and 0 otherwise.
    """
    doc = parse_source(args.input)
    if not doc.sections:
        sys.stderr.write(f"error: no content found in {args.input}\n")
        return 2

    client = LLMClient(
        base_url=args.base_url, model=args.model, timeout=args.timeout,
        retries=args.retries, cache_dir=args.cache_dir,
        debug_dir=args.debug_dir,
    )
    want_summary = not args.no_summary

    # Every stage that calls the model refuses rather than degrades. A
    # document that is half rewritten and half missing would pass into the
    # knowledge graph looking like a complete one.
    try:
        log("stage 1/7  metadata")
        topic = build_metadata(client, doc, want_summary)

        log("stage 2/7  captions")
        fill_missing_captions(client, doc, args.workers)

        log(f"stage 3/7  rewrite ({len(doc.sections)} sections, "
            f"{len(doc.tables)} tables)")
        rewrite_sections(client, doc, args.workers)
        rewrite_tables(client, doc, args.workers)
        rewrite_diagrams(client, doc, args.workers)

        log("stage 4/7  claim splitting")
        split_claims(client, doc, args.workers)

        log("stage 5/7  back-reference repair")
        repair_backreferences(client, doc, args.workers)

        audit: Optional[Dict[str, List[str]]] = None
        if args.no_audit:
            log("stage 6/7  audit skipped (--no-audit)")
        else:
            log("stage 6/7  audit")
            audit = audit_sections(client, doc, args.workers)
    except LLMError as exc:                               # noqa: BLE001
        sys.stderr.write(f"error: {exc}\n")
        sys.stderr.write(f"hint: check the backend at {args.base_url}\n")
        return 3

    log("stage 7/7  render")
    output = render_document(doc, topic, args.model, want_summary)

    problems = verify(output, doc, audit)
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
              f"{len(doc.tables)} tables, {doc.content_words} source words "
              f"-> {doc.rewritten_words} rewritten, "
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
                   help="parallel LLM calls; the rewrite fans out over these")
    p.add_argument("--cache-dir", default=None,
                   help="cache LLM responses here so reruns are cheap")
    p.add_argument("--debug-dir", default=None,
                   help="write every raw prompt+completion here for diagnosis")
    p.add_argument("--offline", action="store_true",
                   help="no longer supported; the body needs the model")
    p.add_argument("--dry-run", action="store_true",
                   help="alias for --offline")
    p.add_argument("--no-audit", action="store_true",
                   help="skip the coverage audit, roughly halving the calls")
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
    if args.offline or args.dry_run:
        # Kept as a flag rather than dropped so the failure explains itself,
        # instead of arriving as an argparse error from a batch script.
        sys.stderr.write(
            "error: --offline is not supported. Every section, table and "
            "diagram is rewritten by the model, so there is no document to "
            "build without a backend.\n")
        return 2
    if not args.input:
        build_parser().error("input file is required (or use --print-schema)")
    if not os.path.exists(args.input):
        sys.stderr.write(f"error: no such file: {args.input}\n")
        return 2
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
