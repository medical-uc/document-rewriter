#!/usr/bin/env python3
"""
rewrite_medical_md.py
=====================

Rewrite slide-derived medical-school markdown (imperfectly converted from PDF)
into a fresh, coherent, tagged markdown study document of roughly the same
length -- while preserving every ``[FIGURE:...]`` embedded in the source.

The source markdown is treated as a *topic reference*: it tells the pipeline
WHAT to write about, not HOW to phrase it. The prose is written fresh, with
the missing connective tissue and context that the slide dump lacks.

Figure captions/descriptions are the one exception: those are rewritten
*conservatively* -- the model may only reorganise the existing description
into flowing sentences, never invent new visual detail.

Pipeline
--------
  1. parse      -- split source into headings/blocks/figures/references
  2. outline    -- one LLM call -> document plan (title, sections, weights)
  3. captions   -- one LLM call per figure -> <cap> + <desc> (conservative)
  4. placement  -- one LLM call -> figure -> section assignment
                   (deterministic lexical fallback if the model misbehaves)
  5. sections   -- one LLM call per section -> fresh prose + key points +
                   definitions + clinical note, with inline [[FIG:id]] anchors
  6. summary    -- one LLM call -> closing synthesis
  7. render     -- assemble the tagged markdown

Backend
-------
An OpenAI-compatible vLLM server:
    base url : http://jupyter-02.aml1.id.iosda.org:8888
    model    : huatuogpt-o1-72b-4bit   (mlx-community/HuatuoGPT-o1-72B-4bit)

HuatuoGPT-o1 is a reasoning model; its "## Thinking / ## Final Response"
scaffolding (and any <think> block) is stripped from every completion.

Output tag vocabulary
---------------------
See TAG_VOCABULARY below and the README table printed by ``--print-schema``.

Usage
-----
    python rewrite_medical_md.py input.md -o output.md
    python rewrite_medical_md.py input.md -o out.md --workers 6 --verbose
    python rewrite_medical_md.py input.md --dry-run -o out.md   # no network
    python rewrite_medical_md.py --print-schema

Zero third-party dependencies (stdlib only).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://jupyter-02.aml1.id.iosda.org:8888"
DEFAULT_MODEL = "huatuogpt-o1-72b-4bit"
MODEL_REPO = "mlx-community/HuatuoGPT-o1-72B-4bit"

DEFAULT_TIMEOUT = 900          # seconds; a 72B 4-bit MLX server is not fast
DEFAULT_RETRIES = 4
DEFAULT_WORKERS = 4

MAXTOK_OUTLINE = 4096
MAXTOK_CAPTION = 2048
MAXTOK_PLACEMENT = 2048
MAXTOK_EXTRAS = 1536
MAXTOK_SUMMARY = 1536

TEMP_OUTLINE = 0.35
TEMP_CAPTION = 0.15            # conservative: stay close to the source text
TEMP_PLACEMENT = 0.10
TEMP_SECTION = 0.55
TEMP_EXTRAS = 0.40
TEMP_SUMMARY = 0.45

MIN_SECTION_WORDS = 110
MAX_SECTION_WORDS = 950

# A section is treated as a generation failure below this fraction of target.
MIN_CONTENT_RATIO = 0.35
MAX_FIGS_PER_SECTION = 6


def section_token_budget(target_words: int, override: int = 0) -> int:
    """Reasoning models spend a lot of tokens before the answer starts.

    Budget = the prose itself (~1.5 tokens/word) plus generous headroom for
    the '## Thinking' block that HuatuoGPT-o1 emits first.
    """
    if override:
        return override
    return int(max(3072, min(10240, target_words * 2.4 + 3000)))

VERBOSE = False


def log(msg: str) -> None:
    if VERBOSE:
        sys.stderr.write("[rewrite] " + msg + "\n")
        sys.stderr.flush()


# --------------------------------------------------------------------------
# Tag vocabulary (documented, comprehensive)
# --------------------------------------------------------------------------

TAG_VOCABULARY: List[Tuple[str, str, str]] = [
    # (tag, where it appears, what it holds)
    ("doc",   "root",              "Wraps the entire generated document."),
    ("meta",  "doc",               "Document-level metadata block."),
    ("title", "meta",              "Document title. Plain text, no '#' marker."),
    ("topic", "meta",              "One-line statement of the subject matter."),
    ("src",   "meta",              "Provenance: source filename, figure count, model, timestamp."),
    ("obj",   "doc",               "Learning objectives."),
    ("goal",  "obj",               "One learning objective. One per element, no bullet marker."),
    ("sec",   "doc",               "One section. Attributes: id, level."),
    ("head",  "sec",               "Section heading. Plain text; depth lives in sec/@level."),
    ("con",   "sec",               "Narrative content prose. May occur several times per "
                                   "section when figures are interleaved into the flow."),
    ("fig",   "sec / doc",         "A figure. Attribute: id. Always contains the literal "
                                   "[FIGURE:id] marker from the source, plus <cap> and <desc>."),
    ("cap",   "fig",               "Short one-sentence caption for the figure."),
    ("desc",  "fig",               "Figure description, restricted to what a sighted reader "
                                   "would lose without the image: layout, colour, arrow "
                                   "direction, what is labelled where. Claims that stay "
                                   "true with the figure deleted are moved into <con>."),
    ("key",   "sec",               "Key points for the section."),
    ("pt",    "key",               "One key point. One per element, no bullet marker."),
    ("def",   "sec",               "Terms introduced by the section."),
    ("term",  "def",               "One definition, phrased as a complete sentence that "
                                   "begins with the term (e.g. 'Thyroglobulin is a ...')."),
    ("clin",  "sec",               "Clinical relevance or correlation note."),
    ("tbl",   "sec",               "A table carried over or reconstructed from the source."),
    ("note",  "sec",               "Aside, caveat, or a machine-readable generation warning "
                                   "(attribute status=\"failed\") when a section could not "
                                   "be written."),
    ("sum",   "doc",               "Closing synthesis of the whole document."),
    ("ref",   "doc",               "Reference list."),
    ("cit",   "ref",               "A single citation."),
]

SCHEMA_NOTES = """
Markdown syntax policy
----------------------
The tags carry the document structure, so structural markdown is not emitted:
  * <title> and <head> hold plain text -- no '#' or '##' markers. Heading depth
    is expressed by the level attribute on <sec>.
  * List items are elements (<goal>, <pt>, <term>, <cit>), not '- ' bullets.
  * Inline emphasis (**bold**, *italic*) is stripped from generated text by
    default. Pass --keep-emphasis to retain it.
Prose style
-----------
Generated prose is held to a fixed style contract, enforced by prompt and then
checked by a deterministic linter with an optional repair pass:
  * every sentence names its own subject (no leading "It/They/These/This/Such",
    no backward-pointing "its"/"their");
  * quantity words stay welded to their substance ("iodine deficiency", never
    "a deficiency in dietary iodine");
  * causation is a verb and the cause is the subject ("Graves' disease causes
    hyperthyroidism", never "is a frequent cause of");
  * enumerations repeat the head noun ("type 1 deiodinase and type 2
    deiodinase");
  * an abbreviation is fixed once as "thyroxine (T4)" and then never alternates.

Content-level markdown INSIDE <con> is preserved: if a section body genuinely
needs a short list or a table, that stays as markdown, because it is part of the
content rather than part of the document skeleton. The literal [FIGURE:id]
markers are likewise never touched.
"""


def print_schema() -> None:
    w1 = max(len(t) for t, _, _ in TAG_VOCABULARY) + 2
    w2 = max(len(p) for _, p, _ in TAG_VOCABULARY) + 2
    print("Output tag vocabulary")
    print("=" * 76)
    print(f"{'TAG'.ljust(w1)}{'PARENT'.ljust(w2)}MEANING")
    print("-" * 76)
    for tag, parent, meaning in TAG_VOCABULARY:
        print(f"{('<' + tag + '>').ljust(w1)}{parent.ljust(w2)}{meaning}")
    print(SCHEMA_NOTES)
    print("Figure blocks always look like:")
    print("    <fig id=\"p5_b2\">")
    print("    [FIGURE:p5_b2]")
    print("    <cap>...</cap>")
    print("    <desc>...</desc>")
    print("    </fig>")


# --------------------------------------------------------------------------
# LLM client (OpenAI-compatible /v1/chat/completions)
# --------------------------------------------------------------------------

class LLMError(RuntimeError):
    pass


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
        base = self.base_url
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    # -- cache ----------------------------------------------------------
    def _cache_path(self, payload: Dict[str, Any]) -> Optional[str]:
        if not self.cache_dir:
            return None
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return os.path.join(self.cache_dir, hashlib.sha256(blob).hexdigest()[:32] + ".json")

    # -- public ---------------------------------------------------------
    def chat(self, system: str, user: str, max_tokens: int = 2048,
             temperature: float = 0.4, tag: str = "") -> str:
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


class DryRunClient(LLMClient):
    """Offline stand-in used by --dry-run to exercise parsing/assembly."""

    def __init__(self, doc: "SourceDoc") -> None:      # noqa: D107
        super().__init__(cache_dir=None)
        self.doc = doc

    def chat_full(self, system: str, user: str, max_tokens: int = 2048,
                  temperature: float = 0.4, tag: str = "") -> Tuple[str, str]:
        return self._stub(tag), ""

    def _stub(self, tag: str) -> str:
        if ".revise" in tag:
            return ""            # offline: exercise the reject path
        kind = tag.split(":", 1)[0]
        if kind == "outline":
            heads = [h for h in self.doc.headings if h][:10] or ["Overview"]
            secs = [
                {"id": f"s{i+1}", "heading": h.title(), "level": 2,
                 "points": [h], "weight": 3}
                for i, h in enumerate(heads)
            ]
            return json.dumps({
                "title": self.doc.title_guess,
                "topic": f"Dry-run plan derived from {len(self.doc.blocks)} source blocks.",
                "objectives": ["(dry run) objective"],
                "sections": secs,
            })
        if kind == "caption":
            fid = tag.split(":", 1)[1] if ":" in tag else ""
            fig = next((f for f in self.doc.figures if f.fid == fid), None)
            raw = fig.raw if fig else ""
            first = re.split(r"(?<=[.!?])\s", raw.strip())[0] if raw else "Figure."
            # Hand back the raw text undivided so the deterministic desc audit
            # is the thing under test offline.
            return f"<<<CAP>>>\n{first}\n<<<DESC>>>\n{raw}\n<<<FACTS>>>\nNONE"
        if kind == "placement":
            return "[]"          # forces the lexical fallback
        if kind == "section":
            sid = tag.split(":", 1)[1] if ":" in tag else "?"
            fids = [f.fid for f in self.doc.figures if f.section_id.startswith(sid)]
            # Deliberately seeded with violations so the revision loop runs.
            body = ("These cells are responsible for producing calcitonin. "
                    "A deficiency during pregnancy can result in cognitive "
                    "impairments in the child. Graves' disease is a frequent cause "
                    "of hyperthyroidism. Type 1 and Type 2 deiodinases catalyze "
                    "outer ring deiodination. " * 8).strip()
            if fids:
                body += "\n\n[[FIG:" + fids[0] + "]]\n\n" + body
            return body
        if kind == "extras":
            # Deliberately exercises the shapes that used to break: a '**bold**'
            # line with no bullet, and a 'term — definition' line.
            return ("<<<KEY>>>\n(dry run) key point one.\n(dry run) key point two.\n"
                    "<<<DEF>>>\n**Thyroglobulin** — A large glycoprotein scaffold.\n"
                    "Colloid is the stored material inside each follicle.\n"
                    "<<<CLIN>>>\n(dry run) clinical note.\n")
        if kind == "summary":
            return "(dry run) summary."
        if kind.startswith("section") and ".revise" in tag:
            return ""
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


def strip_reasoning(text: str) -> str:
    return split_reasoning(text)[0]


# --------------------------------------------------------------------------
# Source parsing
# --------------------------------------------------------------------------

FIG_START_RE = re.compile(r"^\s*>?\s*\[FIGURE:([^\]]+)\]\s*", re.M)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

# Vocabulary that marks a paragraph as "part of a figure description".
DESC_VOCAB = re.compile(
    r"\b(image|figure|diagram|illustrat\w*|depict\w*|shown|shows|showing|label\w*|"
    r"arrow\w*|leader line\w*|positioned|located|situated|top|bottom|left|right|"
    r"corner|center|centre|colou?r\w*|red|blue|green|yellow|purple|pink|gray|grey|"
    r"orange|brown|structure\w*|represent\w*|text|box|boxes|circle\w*|oval|"
    r"rectangul\w*|rectangle\w*|hexagon\w*|background|section|region|panel|column|"
    r"row|vertical\w*|horizontal\w*|layout|connected|points? (?:to|from)|"
    r"description|visual)\b",
    re.I,
)

# Paragraphs that can never belong to a figure description.
HARD_STOP_RE = re.compile(r"^\s*(#{1,6}\s|>\s*\[FIGURE:|•|\*TABLE|\|)")


@dataclass
class Figure:
    fid: str
    raw: str
    order: int
    caption: str = ""
    description: str = ""
    section_id: str = ""
    facts: List[str] = field(default_factory=list)


@dataclass
class SourceBlock:
    heading: str
    text: str
    kind: str = "para"        # para | list | table

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class SourceDoc:
    path: str
    title_guess: str
    headings: List[str]
    blocks: List[SourceBlock]
    figures: List[Figure]
    references: List[str]

    @property
    def prose_words(self) -> int:
        return sum(b.words for b in self.blocks)


def _split_paragraphs(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.split(r"\n\s*\n", text)
    return [p.strip("\n") for p in raw if p.strip()]


def _is_descriptive(par: str) -> bool:
    hits = len(DESC_VOCAB.findall(par))
    if hits >= 2:
        return True
    if hits >= 1 and re.match(r"^\s*([-*+]|\d+\.)\s", par):
        return True
    return False


def _clean_figure_head(par: str, fid: str) -> str:
    par = re.sub(r"^\s*>\s?", "", par, flags=re.M)
    par = par.replace(f"[FIGURE:{fid}]", "", 1)
    return par.strip()


def parse_source(path: str) -> SourceDoc:
    """Split the source markdown into headings, prose blocks, figures, refs."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    paragraphs = _split_paragraphs(text)

    figures: List[Figure] = []
    kept: List[str] = []          # paragraphs that are NOT figure description
    i = 0
    order = 0
    while i < len(paragraphs):
        par = paragraphs[i]
        m = FIG_START_RE.match(par)
        if not m:
            kept.append(par)
            i += 1
            continue

        fid = m.group(1).strip()
        chunks = [_clean_figure_head(par, fid)]
        j = i + 1
        window: List[Tuple[str, bool]] = []
        gap = 0
        while j < len(paragraphs):
            nxt = paragraphs[j]
            if HARD_STOP_RE.match(nxt):
                break
            desc = _is_descriptive(nxt)
            if not desc:
                gap += 1
                if gap > 2:          # too much non-descriptive drift: stop
                    break
            else:
                gap = 0
            window.append((nxt, desc))
            j += 1

        # Trim trailing non-descriptive paragraphs; keep interior ones so that
        # e.g. a bare bullet list sandwiched inside a description survives.
        last_desc = -1
        for k, (_, d) in enumerate(window):
            if d:
                last_desc = k
        consumed = window[: last_desc + 1]
        chunks.extend(p for p, _ in consumed)

        figures.append(Figure(fid=fid, raw="\n\n".join(c for c in chunks if c).strip(),
                              order=order))
        order += 1
        kept.append(f"@@FIGURE_ANCHOR:{fid}@@")
        i = i + 1 + len(consumed)

    # ---- headings, blocks, references --------------------------------
    blocks: List[SourceBlock] = []
    headings: List[str] = []
    references: List[str] = []
    current_heading = ""
    in_refs = False

    for par in kept:
        if par.startswith("@@FIGURE_ANCHOR:"):
            continue
        first = par.splitlines()[0]
        hm = HEADING_RE.match(first)
        if hm and len(par.splitlines()) == 1:
            current_heading = hm.group(2).strip().rstrip(":")
            in_refs = bool(re.search(r"\breferences?\b", current_heading, re.I))
            if not in_refs:
                headings.append(current_heading)
            continue
        if in_refs:
            for line in par.splitlines():
                line = line.strip()
                if re.match(r"^[-*+]\s+", line):
                    references.append(re.sub(r"^[-*+]\s+", "", line).strip())
                elif line:
                    references.append(line)
            in_refs = False
            continue

        kind = "para"
        if par.lstrip().startswith("|") or par.lstrip().startswith("*TABLE"):
            kind = "table"
        elif re.match(r"^\s*([-*+•]|\d+\.)\s", par):
            kind = "list"
        blocks.append(SourceBlock(heading=current_heading, text=par.strip(), kind=kind))

    title_guess = _guess_title(text, headings)
    doc = SourceDoc(path=path, title_guess=title_guess, headings=headings,
                    blocks=blocks, figures=figures, references=references)
    log(f"parsed {os.path.basename(path)}: {len(blocks)} blocks, "
        f"{len(figures)} figures, {len(references)} refs, {doc.prose_words} prose words")
    return doc


def _guess_title(text: str, headings: List[str]) -> str:
    m = re.search(r"^\*\*(.+?)\*\*\s*$", text, re.M)
    if m:
        return m.group(1).strip()
    for h in headings:
        if len(h.split()) >= 2 and not re.match(r"^(references?|functions?)$", h, re.I):
            return h.title()
    return headings[0].title() if headings else "Study Material"


# --------------------------------------------------------------------------
# Lightweight lexical retrieval (stdlib only)
# --------------------------------------------------------------------------

STOPWORDS = set("""a an the and or of in on to for with by is are was were be been being as at from
that this these those it its into which who whom whose what when where how not no than then also
can may might will would should could each other more most such very between within during than
shown show shows image figure diagram""".split())

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS and len(t) > 2]


class Retriever:
    """Tiny TF-IDF ranker over the source blocks."""

    def __init__(self, blocks: Sequence[SourceBlock]) -> None:
        self.blocks = list(blocks)
        self.docs = [tokenize(b.heading + " " + b.text) for b in self.blocks]
        self.df: Dict[str, int] = {}
        for d in self.docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1
        self.n = max(1, len(self.docs))

    def idf(self, term: str) -> float:
        return math.log((self.n + 1) / (self.df.get(term, 0) + 1)) + 1.0

    def score(self, query_tokens: Sequence[str], doc_tokens: Sequence[str]) -> float:
        if not doc_tokens:
            return 0.0
        tf: Dict[str, int] = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for q in set(query_tokens):
            if q in tf:
                s += self.idf(q) * (1 + math.log(tf[q]))
        return s / math.sqrt(len(doc_tokens))

    def top(self, query: str, k: int = 8, char_budget: int = 6000) -> List[SourceBlock]:
        q = tokenize(query)
        scored = [(self.score(q, d), i) for i, d in enumerate(self.docs)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        out: List[SourceBlock] = []
        used = 0
        for s, i in scored:
            if s <= 0 and out:
                break
            b = self.blocks[i]
            if used + len(b.text) > char_budget and out:
                continue
            out.append(b)
            used += len(b.text)
            if len(out) >= k:
                break
        out.sort(key=lambda b: self.blocks.index(b))
        return out


# --------------------------------------------------------------------------
# JSON / delimiter helpers
# --------------------------------------------------------------------------

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


def split_delims(text: str, names: Sequence[str]) -> Dict[str, str]:
    """Parse a `<<<NAME>>>`-delimited response into a dict."""
    pattern = re.compile(r"<{2,3}\s*(" + "|".join(names) + r")\s*>{2,3}", re.I)
    out: Dict[str, str] = {n: "" for n in names}
    matches = list(pattern.finditer(text))
    if not matches:
        out[names[0]] = text.strip()
        return out
    for idx, m in enumerate(matches):
        key = m.group(1).upper()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        out[key] = text[m.end():end].strip()
    return out


# A real markdown bullet is a marker FOLLOWED BY WHITESPACE. Requiring the
# trailing space -- and refusing to treat the first '*' of '**bold**' as a
# marker -- is what stops '**term** - x' from being mangled into '*term** - x'.
BULLET_RE = re.compile(r"^\s*(?:[-+•‣▪]|\*(?!\*)|\d+[.)])\s+")

EMPHASIS_RE = re.compile(r"(\*{1,3}|_{2,3})(?=\S)(.+?)(?<=\S)\1", re.S)
MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)


def strip_bullet(line: str) -> str:
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


def bulletize(text: str, limit: int = 8, keep_emphasis: bool = False) -> List[str]:
    items: List[str] = []
    for line in text.splitlines():
        line = strip_bullet(line)
        if not line:
            continue
        line = clean_inline_md(line, keep_emphasis)
        if line:
            items.append(line)
    return items[:limit]


DEF_SEP_RE = re.compile(r"^(.{1,90}?)\s*(?:—|–|--|::|:)\s+(.+)$", re.S)
_COPULA_RE = re.compile(r"^(is|are|refers?\b|denotes?\b|describes?\b|means?\b)", re.I)


def normalize_definition(line: str, keep_emphasis: bool = False) -> str:
    """Render a definition as a sentence: 'Thyroglobulin is a large ...'.

    Accepts either a full sentence (passed through) or the older
    'term — definition' shape, which is converted.
    """
    s = clean_inline_md(strip_bullet(line), keep_emphasis)
    m = DEF_SEP_RE.match(s)
    if m:
        term, body = m.group(1).strip(), m.group(2).strip()
        # Only de-capitalise a leading article; leave acronyms and proper
        # nouns ('TSH — Thyroid-Stimulating Hormone ...') alone.
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
    return len(re.sub(r"[#*`>|\[\]]", " ", text).split())


# --------------------------------------------------------------------------
# Prose quality: the style contract, the visual/content split, and the linter
# --------------------------------------------------------------------------

STYLE_CONTRACT = """\
WRITING RULES -- every one of these is mandatory.

1. EVERY SENTENCE GETS ITS OWN SUBJECT.
   Never open a sentence with "It", "They", "These", "Those", "This", "That" or
   "Such", and do not point backwards with "its" or "their" when a name is
   available. Name the thing again, even when the repetition feels heavy.
     BAD   These cells are responsible for producing calcitonin.
     GOOD  Parafollicular cells produce calcitonin.
     BAD   Its function is to store hormone.
     GOOD  Colloid stores hormone.

2. NEVER STRAND A MODIFIER FROM ITS NOUN.
   Keep "deficiency", "excess", "insufficient" and "elevated" next to the
   substance they qualify, preferably welded into a compound noun. A reader must
   never have to look at the previous sentence to learn a deficiency of what.
     BAD   A deficiency in dietary iodine can lead to goiter.
     GOOD  Iodine deficiency causes goiter.
     BAD   Hyperthyroidism occurs when there is an excess of thyroid hormones.
     GOOD  Excess thyroid hormone causes hyperthyroidism.
     BAD   A deficiency during pregnancy can result in cognitive impairments.
     GOOD  Thyroid hormone deficiency during pregnancy causes cognitive
           impairment in the child.
   Keep process nouns whole rather than splitting them across a clause:
     BAD   Calcitonin acts by inhibiting bone resorption.
     GOOD  Calcitonin inhibits bone resorption.

3. STATE CAUSATION WITH A VERB, AND LEAD WITH THE CAUSE.
   The subject of the sentence is the cause; the object is the effect; both live
   in the SAME sentence. Prefer "causes", "leads to", "results in",
   "progresses to". Do not write "is a cause of", "the cause is", or "is
   responsible for".
     BAD   Graves' disease is a frequent cause of hyperthyroidism.
     GOOD  Graves' disease causes hyperthyroidism.
     BAD   The most common cause is Hashimoto's thyroiditis.
     GOOD  Hashimoto's thyroiditis is the most common cause of hypothyroidism.
   Name the causal agent precisely rather than gesturing at a process:
     BAD   Autoimmune reactions against thyroid peroxidase contribute to
           Hashimoto's thyroiditis.
     GOOD  Anti-thyroid-peroxidase autoantibodies cause Hashimoto's thyroiditis.

4. REPEAT THE HEAD NOUN IN AN ENUMERATION.
   Do not let two modifiers share one plural noun.
     BAD   Type 1 and Type 2 deiodinases catalyze outer ring deiodination of T4.
     GOOD  Type 1 deiodinase and type 2 deiodinase catalyze the outer ring
           deiodination of T4, producing T3.

5. FIX EACH ABBREVIATION ONCE, THEN DO NOT ALTERNATE.
   Introduce a term as "thyroxine (T4)" on first use, then use ONE form for the
   rest of the document. Never use the expansion and the abbreviation for the
   same referent in a single sentence. Never let a category word stand in for a
   specific molecule: if the sentence is about one molecule, write "T4" or "T3",
   not "thyroid hormone".
"""

VAGUE_UMBRELLA_HINT = (
    "6. DO NOT USE THESE VAGUE UMBRELLA TERMS FOR A SPECIFIC MOLECULE; name the "
    "molecule instead: {terms}.\n"
)

# Inline anchor the section writer emits so figures land in the prose flow.
FIG_MARKER_RE = re.compile(r"^[ \t]*\[?\[FIG:([^\]]+)\]\]?[ \t]*$", re.M)

_ABBREV_GUARD = re.compile(
    r"\b(e\.g|i\.e|etc|vs|cf|approx|ca|Dr|Prof|Fig|No|St|Mr|Mrs|Ms|al|Inc|Ltd)\.", re.I)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[ \t]+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> List[str]:
    """Sentence splitter that survives 'e.g.', decimals and 'T4.'."""
    guarded = _ABBREV_GUARD.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    guarded = re.sub(r"(?<=\d)\.(?=\d)", "\x00", guarded)
    out: List[str] = []
    for line in guarded.split("\n"):
        for part in _SENT_SPLIT_RE.split(line):
            part = part.replace("\x00", ".").strip()
            if part:
                out.append(part)
    return out


# Cues that are visual wherever they appear: depiction verbs, graphical
# primitives, colour, and typography.
DEPICTION_RE = re.compile(r"""
    \b(
      shown|shows|showing|depict\w*|illustrat\w*|display\w*|
      appear\w*|visible|pictured|drawn|draws|render\w*|portray\w*|
      label\w*|labell?ed|annotat\w*|denoted|
      arrow\w*|arrowhead\w*|leader\s+line\w*|dashed|dotted|
      diagram|figure|image|panel|inset|legend|caption|micrograph|photograph|
      schematic|flowchart|illustration|
      background|foreground|
      colou?r\w*|red|blue|green|yellow|purple|pink|orange|brown|gray|grey|
      violet|teal|shaded|shading|hue|
      circular|oval|hexagon\w*|rectangl\w*|square\w*|triangl\w*|
      wavy|curved|coiled|stacked|scattered|outlined|elongated|V-shaped|
      scale\s+bar|tick\s+mark\w*|
      written|font|bold|italic|typeface|
      points?\s+(?:to|from|towards?|upward|downward)
    )\b
""", re.X | re.I)

# Position words are visual ONLY inside a frame-of-reference construction.
# "at the top of the panel" describes the image; "the right lobe" is anatomy
# that stays true with the figure deleted, so it belongs in <con>.
SPATIAL_RE = re.compile(r"""
    (?:
      \b(?:at|on|in|to|toward|towards|from|along|across)\s+
        the\s+(?:top|bottom|left|right|upper|lower|centre|center|middle|
                 far\s+\w+|near\s+\w+)\b
    | \b(?:top|bottom|upper|lower)[\s-](?:left|right|centre|center|half|
        corner|portion|section|panel|row)\b
    | \b(?:left|right|upper|lower|top|bottom|central|middle)\s+
        (?:side|corner|portion|half|panel|column|row|margin|edge|inset)\b
    | \b(?:above|below|beside|alongside|adjacent\s+to|surrounding|next\s+to)\s+
        (?:it|this|the\s+\w+)\b
    | \bof\s+the\s+(?:image|figure|diagram|illustration|panel|photograph)\b
    )
""", re.X | re.I)


def is_visual_sentence(sentence: str) -> bool:
    return bool(DEPICTION_RE.search(sentence) or SPATIAL_RE.search(sentence))


def visual_split(desc: str) -> Tuple[List[str], List[str]]:
    """Partition a description into (visual sentences, content claims).

    The rule, stated by hand: if a sentence can be asserted without saying
    "shown", "depicted", "the arrow", "on the left" or "labelled", then it is
    not a figure description -- it is subject-matter content that belongs in
    <con>, because it stays true when the figure is deleted.
    """
    visual: List[str] = []
    content: List[str] = []
    for sent in split_sentences(desc):
        (visual if is_visual_sentence(sent) else content).append(sent)
    return visual, content


@dataclass
class Violation:
    kind: str
    sentence: str
    detail: str

    def render(self) -> str:
        return f"[{self.kind}] {self.detail}\n    > {self.sentence}"


# -- rule 1: anaphora -------------------------------------------------------
ANAPHOR_START_RE = re.compile(
    r"^\s*(?:And\s+|But\s+|However,\s*|Therefore,\s*|Thus,\s*)?"
    r"(It|They|Them|These|Those|This|That|Such|Its|Their)\b"
    r"(?!\s+(?:is\s+worth|said|being)\b)", re.I)
POSSESSIVE_ANAPHOR_RE = re.compile(r"\b(its|their)\s+([a-z]+)", re.I)

# -- rule 2: stranded modifier ---------------------------------------------
STRANDED_RE = re.compile(
    r"\b(?:a|an|the)\s+(deficiency|excess|insufficiency|elevation|shortage|"
    r"abundance)\b(?!\s+of\s+[a-z])", re.I)
LOOSE_QUANTITY_RE = re.compile(
    r"\b(deficiency|excess|insufficiency|elevation)\s+(?:in|of)\s+"
    r"(?:the\s+|dietary\s+|circulating\s+)?([a-z][\w-]*)", re.I)
THERE_IS_RE = re.compile(
    r"\bthere\s+(?:is|are|'s)\s+(?:an?\s+)?(excess|deficiency|insufficiency|"
    r"elevation|shortage)\b", re.I)

# -- rule 3: nominalised or reversed causation ------------------------------
NOMINAL_CAUSE_RE = re.compile(
    r"\b(is|are|was|were)\s+(?:a|an|one\s+of\s+the)\s+"
    r"(?:\w+\s+){0,2}(cause|causes|contributor|factor)\b", re.I)
REVERSED_CAUSE_RE = re.compile(
    r"\b(?:the\s+)?(?:most\s+common\s+|primary\s+|main\s+|leading\s+|chief\s+)?"
    r"cause(?:s)?\s+(?:is|are|include[s]?|being)\b", re.I)
RESPONSIBLE_RE = re.compile(r"\bis\s+responsible\s+for\b|\bcontribute[s]?\s+to\b", re.I)
# "Calcitonin acts by inhibiting bone resorption" splits the verb from its
# process noun; "Calcitonin inhibits bone resorption" keeps it whole.
SPLIT_PROCESS_RE = re.compile(
    r"\b(act[s]?|function[s]?|work[s]?|operate[s]?)\s+(?:by|through|via)\s+(\w+ing)\b"
    r"|\b(?:serve[s]?|help[s]?)\s+to\s+(\w+)\b"
    r"|\bplay[s]?\s+an?\s+(?:\w+\s+)?role\s+in\s+(\w+ing)\b", re.I)

# -- rule 4: shared head noun in an enumeration -----------------------------
SHARED_HEAD_RE = re.compile(
    r"\b([A-Za-z][\w-]*)\s+(\d+|[IVX]+)\s+and\s+(?:\1\s+)?(\d+|[IVX]+)\s+"
    r"([a-z][\w-]*s)\b", re.I)

# -- rule 5: abbreviation handling ------------------------------------------
ABBR_DEF_RE = re.compile(
    r"\b([A-Za-z][\w'-]*(?:[ -][A-Za-z][\w'-]*){0,4})\s*\(([A-Za-z][A-Za-z0-9]{0,7})\)")


def _plausible_abbreviation(expansion: str, abbr: str) -> bool:
    """Is `abbr` plausibly short for `expansion`?"""
    letters = [c for c in abbr if c.isalpha()]
    if not letters or not any(c.isupper() for c in abbr):
        return False
    words = [w for w in re.split(r"[ -]+", expansion) if w]
    if not words:
        return False
    initials = "".join(w[0] for w in words).lower()
    joined = "".join(letters).lower()
    if joined == initials:
        return True                                   # thyroid-stimulating hormone (TSH)
    if len(letters) == 1 and words and words[-1][0].lower() == letters[0].lower():
        return True                                   # thyroxine (T4)
    if initials.endswith(joined) or joined in initials:
        return True
    # Abbreviations that pick up interior letters: thyroid PerOxidase -> TPO.
    # Require the first letter to anchor on the first word, then match the rest
    # as an in-order subsequence of the expansion's letters.
    flat = "".join(c for c in expansion.lower() if c.isalpha())
    if joined and words[0][0].lower() == joined[0]:
        pos = 0
        for ch in joined:
            pos = flat.find(ch, pos)
            if pos < 0:
                return False
            pos += 1
        return True
    return False


def find_abbreviations(text: str) -> Dict[str, str]:
    """Map abbreviation -> expansion for every 'expansion (ABBR)' definition."""
    pairs: Dict[str, str] = {}
    for m in ABBR_DEF_RE.finditer(text):
        expansion, abbr = m.group(1).strip(), m.group(2).strip()
        if _plausible_abbreviation(expansion, abbr):
            pairs.setdefault(abbr, expansion)
    return pairs


class ProseLinter:
    """Deterministic checks for the writing rules in STYLE_CONTRACT.

    The model is the primary enforcement mechanism -- this class exists to catch
    what the model misses and to feed concrete, quotable violations back into a
    revision pass.
    """

    def __init__(self, vague_terms: Sequence[str] = ()) -> None:
        self.vague_terms = [t.strip().lower() for t in vague_terms if t.strip()]

    # -- public ---------------------------------------------------------
    def lint(self, text: str) -> List[Violation]:
        stripped = FIG_MARKER_RE.sub("", text)
        sentences = split_sentences(stripped)
        out: List[Violation] = []
        for sent in sentences:
            out.extend(self._anaphora(sent))
            out.extend(self._stranded(sent))
            out.extend(self._causation(sent))
            out.extend(self._enumeration(sent))
            out.extend(self._vague(sent))
        out.extend(self._abbreviations(stripped, sentences))
        return out

    # -- rules ----------------------------------------------------------
    def _anaphora(self, s: str) -> List[Violation]:
        v: List[Violation] = []
        m = ANAPHOR_START_RE.match(s)
        if m:
            v.append(Violation(
                "anaphora", s,
                f'Sentence opens with "{m.group(1)}". Name the referent instead.'))
        for pm in POSSESSIVE_ANAPHOR_RE.finditer(s):
            v.append(Violation(
                "anaphora", s,
                f'"{pm.group(0)}" points backwards. Write the owner\'s name.'))
            break
        return v

    def _stranded(self, s: str) -> List[Violation]:
        v: List[Violation] = []
        m = STRANDED_RE.search(s)
        if m:
            v.append(Violation(
                "stranded-modifier", s,
                f'"{m.group(0)}" does not say a {m.group(1)} of what. '
                f'Use a compound noun such as "iodine {m.group(1)}".'))
        m = LOOSE_QUANTITY_RE.search(s)
        if m:
            v.append(Violation(
                "stranded-modifier", s,
                f'"{m.group(0)}" separates the quantity from the substance. '
                f'Write "{m.group(2)} {m.group(1).lower()}".'))
        m = THERE_IS_RE.search(s)
        if m:
            v.append(Violation(
                "stranded-modifier", s,
                f'"{m.group(0)}" buries the subject. Lead with the substance.'))
        return v

    def _causation(self, s: str) -> List[Violation]:
        v: List[Violation] = []
        if NOMINAL_CAUSE_RE.search(s):
            v.append(Violation(
                "nominal-causation", s,
                'Causation stated as a noun ("is a cause of"). Use a verb: '
                '"X causes Y".'))
        if REVERSED_CAUSE_RE.search(s):
            v.append(Violation(
                "reversed-causation", s,
                'The effect is the subject ("the cause is X"). Lead with the '
                'cause: "X causes Y" or "X is the most common cause of Y".'))
        m = RESPONSIBLE_RE.search(s)
        if m:
            v.append(Violation(
                "nominal-causation", s,
                f'"{m.group(0)}" is vague about the causal relation. Use '
                f'"causes", "leads to" or "results in".'))
        m = SPLIT_PROCESS_RE.search(s)
        if m:
            verb = next((g for g in m.groups() if g), "")
            v.append(Violation(
                "split-process", s,
                f'"{m.group(0)}" splits the verb from its process noun. '
                f'Make "{verb}" the main verb.'))
        return v

    def _enumeration(self, s: str) -> List[Violation]:
        m = SHARED_HEAD_RE.search(s)
        if m:
            head = m.group(4)
            singular = head[:-2] if head.endswith("es") and len(head) > 4 else head[:-1]
            return [Violation(
                "shared-head-noun", s,
                f'"{m.group(0)}" makes two modifiers share one plural noun. '
                f'Write "{m.group(1)} {m.group(2)} {singular} and '
                f'{m.group(1).lower()} {m.group(3)} {singular}".')]
        return []

    def _vague(self, s: str) -> List[Violation]:
        low = s.lower()
        for term in self.vague_terms:
            if term in low:
                return [Violation(
                    "vague-umbrella", s,
                    f'"{term}" stands in for a specific molecule. Name it.')]
        return []

    def _abbreviations(self, text: str, sentences: Sequence[str]) -> List[Violation]:
        pairs = find_abbreviations(text)
        out: List[Violation] = []
        for abbr, expansion in pairs.items():
            exp_re = re.compile(r"\b" + re.escape(expansion) + r"\b", re.I)
            abbr_re = re.compile(r"\b" + re.escape(abbr) + r"\b")
            # Both forms in one sentence. The definition site "expansion (ABBR)"
            # is legal and must be blanked WHOLE -- blanking only "(ABBR)" would
            # leave the expansion behind and flag every legitimate definition.
            defsite_re = re.compile(
                r"\b" + re.escape(expansion) + r"\s*\(\s*" + re.escape(abbr) + r"\s*\)",
                re.I)
            for sent in sentences:
                probe = defsite_re.sub(" ", sent)
                if exp_re.search(probe) and abbr_re.search(probe):
                    out.append(Violation(
                        "abbreviation-mixing", sent,
                        f'"{expansion}" and "{abbr}" both name the same thing in '
                        f'one sentence. Keep one form.'))
                    break
                if defsite_re.search(sent) and len(exp_re.findall(probe)) >= 1:
                    out.append(Violation(
                        "abbreviation-mixing", sent,
                        f'"{expansion}" is defined as "{abbr}" and then used in '
                        f'long form again in the same sentence. Switch to '
                        f'"{abbr}" immediately after the definition.'))
                    break
            # Any reuse of the long form after the definition is a violation:
            # once the short form is fixed, one form is used from then on. This
            # fires whether or not the abbreviation is also still in play.
            split = defsite_re.split(text, maxsplit=1)
            if len(split) >= 2:
                after = split[-1]
                reuse = len(exp_re.findall(after))
                if reuse >= 1:
                    out.append(Violation(
                        "abbreviation-alternation", f"{expansion} / {abbr}",
                        f'After defining "{expansion} ({abbr})" the long form is '
                        f'used {reuse} more time(s). Use "{abbr}" from then on.'))
        return out


def summarize_violations(violations: Sequence[Violation]) -> str:
    counts: Dict[str, int] = {}
    for v in violations:
        counts[v.kind] = counts.get(v.kind, 0) + 1
    return ", ".join(f"{k}={n}" for k, n in sorted(counts.items())) or "none"


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

SYS_EDUCATOR = (
    "You are an experienced medical educator and textbook author. You write clear, "
    "accurate, well-organised study material for medical students. You explain "
    "mechanisms rather than listing keywords, you supply the connective context that "
    "lecture slides omit, and you never invent clinical facts you are unsure of. "
    "You write in plain markdown and never wrap your answer in code fences."
)

SYS_CAPTIONER = (
    "You are a scientific copy-editor preparing figure captions. You are given a raw, "
    "machine-generated description of a figure. Your job is to rewrite that raw text "
    "into fluent sentences AND to separate it into two kinds of statement: what a "
    "sighted reader would lose without the image, and subject-matter claims that stay "
    "true with the image deleted. You must not add any visual detail, structure, "
    "label, colour, quantity or relationship that is not already in the raw "
    "description, and you must not delete substantive content: every statement in the "
    "raw text must end up in one bucket or the other. You are re-sorting and "
    "re-phrasing, never summarising and never embellishing."
)

SYS_PLANNER = (
    "You are a curriculum designer structuring a medical study document. "
    "You respond with valid JSON only, no prose, no code fences."
)


def prompt_outline(doc: SourceDoc, target_words: int, max_chars: int) -> str:
    source_view = "\n\n".join(f"[{b.heading}] {b.text}" for b in doc.blocks)
    if len(source_view) > max_chars:
        source_view = source_view[:max_chars] + "\n...[truncated]"

    fig_view = "\n".join(
        f"- {f.fid}: {' '.join(f.raw.split())[:220]}" for f in doc.figures
    ) or "(none)"

    # Enough sections that no single one has to carry a pile of figures.
    n_min = max(6, math.ceil(len(doc.figures) / MAX_FIGS_PER_SECTION))
    plan_hint = f"{n_min} to {max(n_min + 4, 12)} sections"

    return (
        "Below is a fragmented set of lecture-slide notes converted from a PDF. It is "
        "disorganised, full of bare keyword bullets, and missing explanatory context.\n\n"
        "Your task is to plan a FRESH study document on the same subject matter. Use the "
        "notes only to decide WHICH topics belong in the document and in what depth. Do "
        "not plan to copy their wording or their ragged slide-by-slide ordering. Design "
        "the structure a good textbook chapter would use.\n\n"
        "The finished document should total roughly " + str(target_words) + " words of "
        "prose. Plan " + plan_hint + ".\n\n"
        "You must also accommodate the figures listed further below: every one of them "
        "will be placed into one of your sections, so make sure sections exist that "
        "those figures naturally belong to.\n\n"
        "=== SOURCE NOTES ===\n" + source_view + "\n=== END SOURCE NOTES ===\n\n"
        "=== FIGURE INVENTORY ===\n" + fig_view + "\n=== END FIGURE INVENTORY ===\n\n"
        "Respond with JSON of exactly this shape:\n"
        '{"title": "...", "topic": "one sentence", '
        '"objectives": ["...", "..."], '
        '"sections": [{"id": "s1", "heading": "...", "level": 2, '
        '"points": ["specific thing to cover", "..."], "weight": 3, '
        '"wants_definitions": true, "wants_clinical": false}]}\n\n'
        "weight is 1-5 and controls how long that section should be. "
        "points must be concrete content commitments, 2-6 per section. "
        "Output JSON only."
    )


def prompt_caption(fig: Figure, doc_title: str) -> str:
    return (
        f"Document subject: {doc_title}\n"
        f"Figure identifier: {fig.fid}\n\n"
        "Raw machine-generated description:\n"
        "-----\n" + fig.raw + "\n-----\n\n"
        "Produce three things.\n\n"
        "1. CAP -- a caption: ONE sentence naming what the figure shows, derived "
        "strictly from the raw text above.\n\n"
        "2. DESC -- the visual description, and ONLY the visual content: what a "
        "sighted reader would lose by not seeing the image. That means layout, "
        "colour, shape, arrow direction, what is labelled and where, which parts sit "
        "next to which. Write it as flowing prose rather than a numbered inventory.\n\n"
        "   THE TEST: if you can state a sentence without saying \"shown\", "
        "\"depicted\", \"the arrow\", \"on the left\", \"labelled\" or similar, it is "
        "NOT a figure description. It is subject-matter content, and it goes in "
        "FACTS instead.\n"
        "   Anatomical names are not positions: \"the right lobe and the left lobe\" "
        "is anatomy and belongs in FACTS, whereas \"the right lobe is drawn at the "
        "top left of the panel\" is layout and belongs in DESC.\n\n"
        "3. FACTS -- every claim in the raw text that stays true with the image "
        "deleted: mechanisms, quantities, anatomical relationships, what a structure "
        "does. One claim per line, no bullet characters. Write each as a standalone "
        "sentence that names its own subject, because these lines are handed to a "
        "writer who cannot see the figure. Write \"Iodide is concentrated about "
        "30-fold above plasma\", never \"it is concentrated 30-fold\".\n"
        "   Do not invent claims. If the raw text is purely visual, write NONE.\n\n"
        "Every statement in the raw text must appear in DESC or in FACTS. Do not "
        "drop anything and do not put the same statement in both.\n\n"
        "Where the raw text says \"it\", \"they\", \"these cells\" or \"this "
        "structure\", replace the pronoun with the name it refers to, using only "
        "names already present in the raw text above.\n\n"
        "Respond in exactly this format:\n"
        "<<<CAP>>>\n(one sentence)\n<<<DESC>>>\n(visual prose)\n<<<FACTS>>>\n"
        "(one claim per line, or NONE)"
    )


def prompt_placement(outline: "Outline", figures: Sequence[Figure]) -> str:
    secs = "\n".join(
        f"- {s.sid}: {s.heading} :: " + "; ".join(s.points) for s in outline.sections
    )
    figs = "\n".join(
        f"- {f.fid}: {f.caption or ' '.join(f.raw.split())[:200]}" for f in figures
    )
    return (
        "You are placing figures into a study document.\n\n"
        "=== SECTIONS ===\n" + secs + "\n\n"
        "=== FIGURES ===\n" + figs + "\n\n"
        "Assign EVERY figure to exactly one section, choosing the section whose subject "
        "matter the figure actually illustrates. A section may receive several figures; "
        "a section may receive none.\n\n"
        'Respond with JSON only: [{"fid": "...", "sid": "..."}, ...]'
    )


def prompt_section_prose(
    outline: "Outline",
    sec: "PlannedSection",
    excerpts: Sequence[SourceBlock],
    figs: Sequence[Figure],
    target_words: int,
    insist: bool = False,
    style: str = "",
) -> str:
    """Prose-only prompt.

    Deliberately asks for ONE thing. An earlier version asked for prose plus
    key points plus definitions plus a clinical note in a single delimited
    response; the model reliably filled the three short parts and left the long
    one empty. Splitting the call fixed that.
    """
    src = "\n\n".join(f"- {b.text}" for b in excerpts) or "(no closely matching notes)"
    if figs:
        figline = "\n".join(f"[[FIG:{f.fid}]] -- {f.caption}" for f in figs)
        fig_instr = (
            "\nThis section carries the following figures:\n" + figline + "\n\n"
            "Place each marker on its own line at the point in your prose where that "
            "figure belongs, and refer to it naturally in the surrounding sentences. "
            "Use each marker exactly once. Do not put a marker in the first or last "
            "line of the section.\n"
        )
    else:
        fig_instr = ""

    facts: List[str] = []
    for f in figs:
        facts.extend(f.facts)
    if facts:
        fact_block = (
            "\n=== CLAIMS TAKEN OUT OF THIS SECTION'S FIGURE DESCRIPTIONS ===\n"
            + "\n".join(f"- {x}" for x in _dedupe_facts(facts))
            + "\n=== END CLAIMS ===\n\n"
            "These claims were removed from the figure descriptions because they "
            "stay true with the figures deleted. Work them into your prose. Do not "
            "write them back as descriptions of the figures.\n"
        )
    else:
        fact_block = ""

    others = "; ".join(s.heading for s in outline.sections if s.sid != sec.sid)

    urgency = ""
    if insist:
        urgency = (
            "\nIMPORTANT: your previous attempt returned no usable text. Do not plan, "
            "do not comment on the task, do not restate these instructions. Begin your "
            "reply with the first sentence of the section itself.\n"
        )

    return (
        f"Document: {outline.title}\n"
        f"Subject: {outline.topic}\n"
        f"Other sections (do not duplicate their material): {others}\n\n"
        f"Write the body of the section titled: {sec.heading}\n\n"
        "It must cover:\n" + "\n".join(f"- {p}" for p in sec.points) + "\n\n"
        f"Target length: about {target_words} words. This is the single most "
        "important requirement: the section must be substantial, not a summary.\n\n"
        "=== REFERENCE NOTES (fragmentary slide material on this topic) ===\n"
        + src + "\n=== END REFERENCE NOTES ===\n\n" + fact_block + style + "\n"
        "The reference notes tell you WHAT this section should cover. They are a "
        "keyword dump with missing context; do not paraphrase them line by line and do "
        "not reproduce their bullet style. Write fresh, explanatory, connected prose "
        "that a student can actually learn from: state mechanisms, give the reason "
        "behind each fact, define terms on first use, and include the numbers and "
        "specifics the notes mention where they are correct.\n\n"
        "Write in full paragraphs of continuous prose. Do not answer with a bullet "
        "list. Do not write a heading. Do not add a summary, key points, or a list of "
        "definitions -- those are collected separately.\n"
        + fig_instr + urgency +
        "\nOutput the section body and nothing else."
    )


def prompt_section_extras(sec: "PlannedSection", content: str,
                          style: str = "") -> str:
    body = " ".join(content.split())[:5000]
    want_def = ("3 to 5 definitions" if sec.wants_definitions else "0 to 3 definitions")
    return (
        f"Here is a finished section of a medical study document, titled "
        f"\"{sec.heading}\":\n\n{body}\n\n"
        "Produce three short items about THIS text only.\n\n"
        "Respond in exactly this format, keeping every marker:\n"
        "<<<KEY>>>\n"
        "3 to 5 take-home points, one per line, no bullet characters. Each must be a "
        "complete sentence. Do not simply copy sentences from the text.\n"
        "<<<DEF>>>\n"
        f"{want_def}, one per line, no bullet characters. Each MUST be a complete "
        "sentence that begins with the term being defined, in this style:\n"
        "Thyroglobulin is a large glycoprotein that acts as the scaffold for thyroid "
        "hormone synthesis and storage.\n"
        "Do not use dashes, colons, or bold markers to separate the term from its "
        "definition, and do not use any markdown formatting.\n"
        "<<<CLIN>>>\n"
        "1 to 3 sentences on clinical relevance, or the single word NONE.\n\n"
        + style
    )


def prompt_revision(text: str, violations: Sequence[Violation], style: str,
                    has_markers: bool) -> str:
    listed = "\n\n".join(v.render() for v in violations[:14])
    marker_rule = (
        "\nThe text contains figure anchors of the form [[FIG:xxx]]. Reproduce every "
        "one of them, unchanged, on its own line, in the same relative position.\n"
        if has_markers else "")
    return (
        "Revise the passage below so that it obeys the writing rules. The passage is "
        "otherwise correct: keep the same facts, the same order of ideas and "
        "approximately the same length. Change only what the rules require.\n\n"
        "=== PASSAGE ===\n" + text + "\n=== END PASSAGE ===\n\n"
        "=== PROBLEMS FOUND ===\n" + listed + "\n=== END PROBLEMS ===\n\n"
        + style + marker_rule +
        "\nWhere a sentence points backwards with a pronoun, resolve it to the name it "
        "refers to. Where a modifier is stranded, pull its noun in. Where causation is "
        "stated as a noun, restate it with a causal verb and lead with the cause.\n\n"
        "Output the revised passage and nothing else. Do not comment on the changes."
    )


def prompt_summary(outline: "Outline", sections: Sequence["WrittenSection"],
                   style: str = "") -> str:
    digest = "\n\n".join(
        f"## {s.heading}\n" + " ".join(s.content.split())[:700] for s in sections
    )
    return (
        f"Document: {outline.title}\n\n"
        "Here is a digest of the finished document:\n\n" + digest + "\n\n"
        "Write a closing synthesis of 120-180 words that ties the material together: "
        "the through-line of the topic, how the pieces relate, and what a student should "
        "retain. Prose only, no headings, no bullet list.\n\n" + style
    )


# --------------------------------------------------------------------------
# Plan / result structures
# --------------------------------------------------------------------------

@dataclass
class PlannedSection:
    sid: str
    heading: str
    level: int
    points: List[str]
    weight: float
    wants_definitions: bool = True
    wants_clinical: bool = False
    target_words: int = 300
    figures: List[Figure] = field(default_factory=list)


@dataclass
class Outline:
    title: str
    topic: str
    objectives: List[str]
    sections: List[PlannedSection]


@dataclass
class WrittenSection:
    sid: str
    heading: str
    level: int
    content: str
    key_points: List[str]
    definitions: List[str]
    clinical: str
    figures: List[Figure]
    failed: bool = False
    planned_points: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Stage 2: outline
# --------------------------------------------------------------------------

def build_outline(client: LLMClient, doc: SourceDoc, target_words: int,
                  max_chars: int) -> Outline:
    raw = client.chat(
        SYS_PLANNER,
        prompt_outline(doc, target_words, max_chars),
        max_tokens=MAXTOK_OUTLINE,
        temperature=TEMP_OUTLINE,
        tag="outline",
    )
    try:
        data = extract_json(raw)
    except ValueError:
        log("outline JSON unparseable; falling back to source headings")
        data = _fallback_outline(doc)

    sections: List[PlannedSection] = []
    for i, s in enumerate(data.get("sections") or []):
        heading = str(s.get("heading") or "").strip()
        if not heading:
            continue
        sections.append(PlannedSection(
            sid=str(s.get("id") or f"s{i + 1}").strip(),
            heading=heading,
            level=int(s.get("level") or 2),
            points=[str(p).strip() for p in (s.get("points") or []) if str(p).strip()],
            weight=float(s.get("weight") or 3),
            wants_definitions=bool(s.get("wants_definitions", True)),
            wants_clinical=bool(s.get("wants_clinical", False)),
        ))
    if not sections:
        data = _fallback_outline(doc)
        sections = [
            PlannedSection(sid=s["id"], heading=s["heading"], level=2,
                           points=s["points"], weight=3)
            for s in data["sections"]
        ]

    # deduplicate ids
    seen: Dict[str, int] = {}
    for s in sections:
        if s.sid in seen:
            seen[s.sid] += 1
            s.sid = f"{s.sid}_{seen[s.sid]}"
        else:
            seen[s.sid] = 0

    total_weight = sum(s.weight for s in sections) or 1.0
    for s in sections:
        raw_target = target_words * (s.weight / total_weight)
        s.target_words = int(max(MIN_SECTION_WORDS, min(MAX_SECTION_WORDS, raw_target)))

    outline = Outline(
        title=clean_inline_md(str(data.get("title") or doc.title_guess)),
        topic=clean_inline_md(str(data.get("topic") or "")),
        objectives=[clean_inline_md(str(o)) for o in (data.get("objectives") or [])
                    if str(o).strip()],
        sections=sections,
    )
    log(f"outline: {len(sections)} sections, "
        f"{sum(s.target_words for s in sections)} planned words")
    return outline


def _fallback_outline(doc: SourceDoc) -> Dict[str, Any]:
    heads: List[str] = []
    for h in doc.headings:
        if h and h.lower() not in {x.lower() for x in heads}:
            heads.append(h)
    heads = heads[:10] or ["Overview"]
    return {
        "title": doc.title_guess,
        "topic": doc.title_guess,
        "objectives": [],
        "sections": [
            {"id": f"s{i + 1}", "heading": h.title(), "points": [h], "weight": 3}
            for i, h in enumerate(heads)
        ],
    }


# --------------------------------------------------------------------------
# Stage 3: figure captions
# --------------------------------------------------------------------------

def write_captions(client: LLMClient, doc: SourceDoc, title: str, workers: int,
                   keep_emphasis: bool = False, audit: bool = True) -> None:
    def one(fig: Figure) -> None:
        try:
            raw = client.chat(
                SYS_CAPTIONER,
                prompt_caption(fig, title),
                max_tokens=MAXTOK_CAPTION,
                temperature=TEMP_CAPTION,
                tag=f"caption:{fig.fid}",
            )
            parts = split_delims(raw, ["CAP", "DESC", "FACTS"])
            cap = " ".join(parts["CAP"].split()).strip()
            desc = parts["DESC"].strip()
            facts = [f for f in bulletize(parts["FACTS"], limit=20,
                                          keep_emphasis=keep_emphasis)
                     if f.strip().upper().rstrip(".") != "NONE"]
        except Exception as exc:                          # noqa: BLE001
            log(f"caption failed for {fig.fid} ({exc}); using raw description")
            cap, desc, facts = "", "", []
        if not desc:
            desc = fig.raw
        if not cap:
            cap = _first_sentence(fig.raw)

        # Deterministic backstop for the model's split. Any sentence left in the
        # description that carries no visual cue is content, so it moves to the
        # facts bucket where the section writer will pick it up.
        if audit:
            visual, content = visual_split(desc)
            if visual and content:
                log(f"desc audit {fig.fid}: moved {len(content)} non-visual "
                    f"sentence(s) into <con>")
                desc = " ".join(visual)
                facts.extend(content)
            elif not visual:
                log(f"desc audit {fig.fid}: no visual sentences found, "
                    f"keeping description unchanged")

        # Figure text follows the same markdown policy as the rest of the
        # document; stripping emphasis changes no content.
        fig.caption = clean_inline_md(cap, keep_emphasis)
        fig.description = clean_inline_md(desc, keep_emphasis)
        fig.facts = _dedupe_facts(clean_inline_md(f, keep_emphasis) for f in facts)

    _run_parallel(one, doc.figures, workers, "captions")
    total = sum(len(f.facts) for f in doc.figures)
    log(f"captions: extracted {total} content claim(s) out of the figure text")


def _dedupe_facts(facts: Iterable[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for f in facts:
        f = f.strip()
        key = re.sub(r"[^a-z0-9 ]", "", f.lower())
        if len(f.split()) < 4 or key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def _first_sentence(text: str) -> str:
    flat = " ".join(text.split())
    m = re.match(r"(.{0,240}?[.!?])(\s|$)", flat)
    return (m.group(1) if m else flat[:200]).strip()


# --------------------------------------------------------------------------
# Stage 4: figure placement
# --------------------------------------------------------------------------

def place_figures(client: LLMClient, outline: Outline, doc: SourceDoc,
                  max_per_section: int = MAX_FIGS_PER_SECTION) -> None:
    valid = {s.sid for s in outline.sections}
    assignment: Dict[str, str] = {}

    if doc.figures:
        try:
            raw = client.chat(
                SYS_PLANNER,
                prompt_placement(outline, doc.figures),
                max_tokens=MAXTOK_PLACEMENT,
                temperature=TEMP_PLACEMENT,
                tag="placement",
            )
            data = extract_json(raw)
            if isinstance(data, dict):
                data = data.get("assignments") or data.get("placements") or []
            for item in data or []:
                if not isinstance(item, dict):
                    continue
                fid = str(item.get("fid") or item.get("figure") or "").strip()
                sid = str(item.get("sid") or item.get("section") or "").strip()
                if fid and sid in valid:
                    assignment[fid] = sid
        except Exception as exc:                          # noqa: BLE001
            log(f"placement call unusable ({exc}); using lexical fallback for all figures")

    # Lexical fallback for anything the model missed or mis-assigned.
    ranker = Retriever([SourceBlock(heading=s.heading,
                                    text=s.heading + " " + " ".join(s.points))
                        for s in outline.sections])
    for fig in doc.figures:
        sid = assignment.get(fig.fid)
        if sid not in valid:
            sid = _best_section(ranker, outline, fig)
            log(f"figure {fig.fid} -> {sid} (lexical fallback)")
        fig.section_id = sid

    by_sid = {s.sid: s for s in outline.sections}
    for fig in sorted(doc.figures, key=lambda f: f.order):
        by_sid[fig.section_id].figures.append(fig)

    _rebalance(ranker, outline, max_per_section)

    log("figure placement: " + ", ".join(
        f"{s.sid}={len(s.figures)}" for s in outline.sections if s.figures))


def _rebalance(ranker: Retriever, outline: Outline, max_per_section: int) -> None:
    """Move overflow figures out of crowded sections.

    A section carrying a dozen figures produces a prompt with a dozen
    [[FIG:]] markers to thread through the prose, which measurably degrades
    the section's own text. Overflow goes to the next-best-scoring section
    that still has room.
    """
    if max_per_section <= 0:
        return
    for sec in outline.sections:
        if len(sec.figures) <= max_per_section:
            continue
        scored = sorted(
            sec.figures,
            key=lambda f: ranker.score(tokenize(f.caption + " " + f.raw),
                                       ranker.docs[_index_of(outline, sec.sid)]),
            reverse=True,
        )
        sec.figures = scored[:max_per_section]
        for fig in scored[max_per_section:]:
            target = _next_best(ranker, outline, fig, exclude=sec.sid,
                                max_per_section=max_per_section)
            fig.section_id = target.sid
            target.figures.append(fig)
            log(f"figure {fig.fid} moved {sec.sid} -> {target.sid} (crowding)")
    for sec in outline.sections:
        sec.figures.sort(key=lambda f: f.order)


def _index_of(outline: Outline, sid: str) -> int:
    for i, s in enumerate(outline.sections):
        if s.sid == sid:
            return i
    return 0


def _next_best(ranker: Retriever, outline: Outline, fig: Figure, exclude: str,
               max_per_section: int) -> PlannedSection:
    query = tokenize(fig.caption + " " + fig.raw)
    ranked = sorted(
        range(len(outline.sections)),
        key=lambda i: ranker.score(query, ranker.docs[i]),
        reverse=True,
    )
    for i in ranked:
        cand = outline.sections[i]
        if cand.sid != exclude and len(cand.figures) < max_per_section:
            return cand
    for i in ranked:
        if outline.sections[i].sid != exclude:
            return outline.sections[i]
    return outline.sections[0]


def _best_section(ranker: Retriever, outline: Outline, fig: Figure) -> str:
    query = tokenize(fig.caption + " " + fig.raw)
    best_i, best_s = 0, -1.0
    for i, dtoks in enumerate(ranker.docs):
        s = ranker.score(query, dtoks)
        if s > best_s:
            best_i, best_s = i, s
    return outline.sections[best_i].sid


# --------------------------------------------------------------------------
# Stage 5: section writing
# --------------------------------------------------------------------------

def _salvage_from_thinking(thinking: str, target_words: int) -> str:
    """Recover prose that the model drafted inside its reasoning block."""
    if not thinking:
        return ""
    paras = [p.strip() for p in re.split(r"\n\s*\n", thinking) if p.strip()]
    # Keep the tail: reasoning models plan first and draft last.
    keep: List[str] = []
    for p in reversed(paras):
        if re.match(r"^(okay|ok|let me|i need to|first,|so,|now i|the user|hmm)", p,
                    re.I):
            continue
        if len(p.split()) < 25:
            continue
        keep.insert(0, p)
        if sum(len(x.split()) for x in keep) >= target_words * 0.8:
            break
    return "\n\n".join(keep).strip()


def revise_prose(client: LLMClient, text: str, linter: ProseLinter, style: str,
                 max_rounds: int, budget: int, tag: str) -> Tuple[str, List[Violation]]:
    """Iteratively repair style violations, never at the cost of content.

    A revision is rejected outright if it drops a figure anchor or loses more
    than a quarter of the text -- a cleaner passage that has silently shed a
    figure or a paragraph is a worse outcome than a passage with violations.
    """
    best = text
    best_v = linter.lint(text)
    if not best_v or max_rounds <= 0:
        return best, best_v

    markers = set(FIG_MARKER_RE.findall(text))
    for rnd in range(1, max_rounds + 1):
        try:
            cand = client.chat(
                SYS_EDUCATOR,
                prompt_revision(best, best_v, style, bool(markers)),
                max_tokens=budget,
                temperature=0.3,
                tag=f"{tag}.revise{rnd}",
            )
        except Exception as exc:                          # noqa: BLE001
            log(f"revision {tag} round {rnd} errored: {exc}")
            break
        cand = _strip_stray_headings(clean_inline_md(cand, keep_emphasis=True))
        if not cand:
            break
        if set(FIG_MARKER_RE.findall(cand)) != markers:
            log(f"revision {tag} round {rnd} rejected: figure anchors changed")
            break
        if word_count(cand) < word_count(best) * 0.75:
            log(f"revision {tag} round {rnd} rejected: lost "
                f"{word_count(best) - word_count(cand)} words")
            break
        cand_v = linter.lint(cand)
        if len(cand_v) >= len(best_v):
            log(f"revision {tag} round {rnd}: no improvement "
                f"({len(best_v)} -> {len(cand_v)}), keeping previous")
            break
        log(f"revision {tag} round {rnd}: violations {len(best_v)} -> {len(cand_v)}")
        best, best_v = cand, cand_v
        if not best_v:
            break
    return best, best_v


def write_sections(client: LLMClient, outline: Outline, doc: SourceDoc,
                   workers: int, keep_emphasis: bool, tok_override: int,
                   failures: List[str], linter: ProseLinter, style: str,
                   max_revisions: int,
                   lint_log: Optional[List[Tuple[str, List[Violation]]]] = None,
                   ) -> List[WrittenSection]:
    retriever = Retriever(doc.blocks)
    results: Dict[str, WrittenSection] = {}

    def get_prose(sec: PlannedSection, excerpts: Sequence[SourceBlock]) -> str:
        budget = section_token_budget(sec.target_words, tok_override)
        floor = max(60, int(sec.target_words * MIN_CONTENT_RATIO))
        thinking = ""

        for attempt in (1, 2):
            try:
                content, thinking = client.chat_full(
                    SYS_EDUCATOR,
                    prompt_section_prose(outline, sec, excerpts, sec.figures,
                                         sec.target_words, insist=(attempt == 2),
                                         style=style),
                    max_tokens=budget if attempt == 1 else int(budget * 1.4),
                    temperature=TEMP_SECTION if attempt == 1 else 0.7,
                    tag=f"section:{sec.sid}" + ("" if attempt == 1 else ".retry"),
                )
            except Exception as exc:                      # noqa: BLE001
                log(f"section {sec.sid} attempt {attempt} errored: {exc}")
                continue

            content = _strip_stray_headings(content)
            if word_count(content) >= floor:
                return content
            log(f"section {sec.sid} attempt {attempt}: only "
                f"{word_count(content)} words (floor {floor}); retrying")

        salvaged = _salvage_from_thinking(thinking, sec.target_words)
        if word_count(salvaged) >= floor:
            log(f"section {sec.sid}: recovered {word_count(salvaged)} words "
                f"from the reasoning block")
            return _strip_stray_headings(salvaged)
        return ""

    def one(sec: PlannedSection) -> None:
        query = sec.heading + " " + " ".join(sec.points)
        excerpts = retriever.top(query, k=10, char_budget=7000)

        content = get_prose(sec, excerpts)
        if content:
            content, viol = revise_prose(
                client, content, linter, style, max_revisions,
                section_token_budget(sec.target_words, tok_override),
                f"section:{sec.sid}")
            if lint_log is not None:
                lint_log.append((f"{sec.sid} ({sec.heading})", viol))
        failed = not content
        if failed:
            failures.append(f"{sec.sid} ({sec.heading})")
            log(f"section {sec.sid}: FAILED to generate prose")

        key: List[str] = []
        defs: List[str] = []
        clin = ""
        if content:
            try:
                raw = client.chat(
                    SYS_EDUCATOR,
                    prompt_section_extras(sec, content, style),
                    max_tokens=MAXTOK_EXTRAS,
                    temperature=TEMP_EXTRAS,
                    tag=f"extras:{sec.sid}",
                )
                parts = split_delims(raw, ["KEY", "DEF", "CLIN"])
                key = bulletize(parts["KEY"], limit=6, keep_emphasis=keep_emphasis)
                if lint_log is not None:
                    extras_v = linter.lint(" ".join(key + [parts["CLIN"]]))
                    if extras_v:
                        lint_log.append((f"{sec.sid} extras", extras_v))
                defs = [normalize_definition(d, keep_emphasis)
                        for d in bulletize(parts["DEF"], limit=5, keep_emphasis=True)]
                defs = [d for d in defs if len(d.split()) >= 4]
                clin = clean_inline_md(" ".join(parts["CLIN"].split()), keep_emphasis)
                if clin.strip().upper().rstrip(".") == "NONE":
                    clin = ""
            except Exception as exc:                      # noqa: BLE001
                log(f"extras for {sec.sid} failed: {exc}")

        results[sec.sid] = WrittenSection(
            sid=sec.sid,
            heading=clean_inline_md(sec.heading, keep_emphasis),
            level=sec.level,
            content=clean_inline_md(content, keep_emphasis) if content else "",
            key_points=key,
            definitions=defs,
            clinical=clin,
            figures=list(sec.figures),
            failed=failed,
            planned_points=list(sec.points),
        )

    _run_parallel(one, outline.sections, workers, "sections")
    return [results[s.sid] for s in outline.sections if s.sid in results]


def _strip_stray_headings(text: str) -> str:
    """The model is told not to emit a heading; remove one if it did anyway."""
    lines = text.splitlines()
    while lines and (not lines[0].strip() or HEADING_RE.match(lines[0].strip())):
        if lines[0].strip() and HEADING_RE.match(lines[0].strip()):
            lines.pop(0)
            continue
        if not lines[0].strip():
            lines.pop(0)
            continue
        break
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------
# Parallel helper
# --------------------------------------------------------------------------

def _run_parallel(fn, items: Sequence[Any], workers: int, label: str) -> None:
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
# Stage 7: rendering
# --------------------------------------------------------------------------

def render_figure(fig: Figure) -> str:
    cap = " ".join((fig.caption or "").split()) or "Figure."
    desc = (fig.description or fig.raw).strip()
    return (
        f'<fig id="{fig.fid}">\n'
        f"[FIGURE:{fig.fid}]\n"
        f"<cap>{cap}</cap>\n"
        f"<desc>\n{desc}\n</desc>\n"
        f"</fig>"
    )


def render_section(sec: WrittenSection) -> str:
    out: List[str] = [f'<sec id="{sec.sid}" level="{sec.level}">',
                      f"<head>{sec.heading}</head>", ""]

    if sec.failed:
        out.append('<note status="failed">')
        out.append("This section could not be generated. Planned coverage: "
                   + "; ".join(sec.planned_points) + ".")
        out.append("</note>")
        out.append("")

    by_id = {f.fid: f for f in sec.figures}
    placed: List[str] = []

    # Split the prose at [[FIG:id]] anchors so figures land in the flow.
    pieces = FIG_MARKER_RE.split(sec.content)
    # split() with one capture group yields: text, fid, text, fid, text...
    chunk_texts = pieces[0::2]
    chunk_fids = pieces[1::2]

    for i, text in enumerate(chunk_texts):
        text = text.strip()
        if text:
            out.append("<con>")
            out.append(text)
            out.append("</con>")
            out.append("")
        if i < len(chunk_fids):
            fid = chunk_fids[i].strip()
            if fid in by_id and fid not in placed:
                out.append(render_figure(by_id[fid]))
                out.append("")
                placed.append(fid)

    # Any figure the model failed to anchor goes after the prose.
    for f in sec.figures:
        if f.fid not in placed:
            out.append(render_figure(f))
            out.append("")
            placed.append(f.fid)

    if sec.key_points:
        out.append("<key>")
        out.extend(f"<pt>{k}</pt>" for k in sec.key_points)
        out.append("</key>")
        out.append("")
    if sec.definitions:
        out.append("<def>")
        out.extend(f"<term>{d}</term>" for d in sec.definitions)
        out.append("</def>")
        out.append("")
    if sec.clinical:
        out.append("<clin>")
        out.append(sec.clinical)
        out.append("</clin>")
        out.append("")

    out.append("</sec>")
    return "\n".join(out).replace("\n\n\n", "\n\n")


def render_document(outline: Outline, sections: Sequence[WrittenSection],
                    doc: SourceDoc, summary: str, model: str,
                    orphans: Sequence[Figure]) -> str:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    parts: List[str] = ["<doc>", ""]

    parts += [
        "<meta>",
        f"<title>{outline.title}</title>",
        f"<topic>{outline.topic}</topic>",
        f"<src>source: {os.path.basename(doc.path)} | figures: {len(doc.figures)} | "
        f"model: {model} | generated: {stamp}</src>",
        "</meta>",
        "",
    ]

    if outline.objectives:
        parts.append("<obj>")
        parts.extend(f"<goal>{o}</goal>" for o in outline.objectives)
        parts.append("</obj>")
        parts.append("")

    for sec in sections:
        parts.append(render_section(sec))
        parts.append("")

    if orphans:
        parts.append('<sec id="figures-appendix" level="2">')
        parts.append("<head>Additional Figures</head>")
        parts.append("")
        parts.append("<con>")
        parts.append("Figures from the source material that did not attach to a "
                     "specific section above are collected here for completeness.")
        parts.append("</con>")
        parts.append("")
        for f in orphans:
            parts.append(render_figure(f))
            parts.append("")
        parts.append("</sec>")
        parts.append("")

    if summary:
        parts += ["<sum>", summary.strip(), "</sum>", ""]

    if doc.references:
        parts.append("<ref>")
        parts.extend(f"<cit>{r}</cit>" for r in doc.references)
        parts.append("</ref>")
        parts.append("")

    parts.append("</doc>")
    text = "\n".join(parts)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def _unused_facts(doc: SourceDoc, sections: Sequence[WrittenSection]) -> List[str]:
    """Claims pulled out of figure descriptions that never reached the prose.

    Moving content from <desc> into <con> only works if the section writer
    actually uses it, so anything dropped on the floor is reported rather than
    silently lost.
    """
    prose = " ".join(s.content for s in sections).lower()
    prose_tokens = set(tokenize(prose))
    missing: List[str] = []
    for fig in doc.figures:
        for fact in fig.facts:
            toks = set(tokenize(fact))
            if not toks:
                continue
            overlap = len(toks & prose_tokens) / len(toks)
            if overlap < 0.5:
                missing.append(f"{fig.fid}: {fact}")
    return missing


def _write_lint_report(path: str, lint_log: Sequence[Tuple[str, List[Violation]]],
                       orphan_facts: Sequence[str]) -> None:
    lines: List[str] = ["STYLE LINT REPORT", "=" * 72, ""]
    total = 0
    for label, viols in lint_log:
        if not viols:
            continue
        total += len(viols)
        lines.append(f"## {label} -- {len(viols)} violation(s)")
        lines.extend(v.render() for v in viols)
        lines.append("")
    lines.insert(2, f"residual violations: {total}")
    if orphan_facts:
        lines.append("## Claims removed from <desc> that never reached <con>")
        lines.extend(f"  - {f}" for f in orphan_facts)
        lines.append("")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError as exc:                                # noqa: BLE001
        log(f"could not write lint report: {exc}")


def verify(output: str, doc: SourceDoc) -> List[str]:
    problems: List[str] = []
    for fig in doc.figures:
        n = output.count(f"[FIGURE:{fig.fid}]")
        if n == 0:
            problems.append(f"MISSING figure marker: {fig.fid}")
        elif n > 1:
            problems.append(f"DUPLICATED figure marker ({n}x): {fig.fid}")
    for tag in ("doc", "meta", "ref"):
        if f"<{tag}>" in output and output.count(f"<{tag}>") != output.count(f"</{tag}>"):
            problems.append(f"unbalanced <{tag}> tags")
    for tag in ("sec", "fig"):
        if output.count(f"<{tag} ") != output.count(f"</{tag}>"):
            problems.append(f"unbalanced <{tag}> tags")
    return problems


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    doc = parse_source(args.input)

    if not doc.blocks:
        sys.stderr.write("error: no prose content found in input\n")
        return 2

    linter = ProseLinter(vague_terms=args.vague_term)
    style = STYLE_CONTRACT
    if args.vague_term:
        style += VAGUE_UMBRELLA_HINT.format(terms=", ".join(
            f'"{t}"' for t in args.vague_term))

    # Content claims migrate out of the figure descriptions into the prose, so
    # the prose target has to grow to absorb them.
    fig_words = sum(len(f.raw.split()) for f in doc.figures)
    target = args.target_words or int(
        (doc.prose_words + fig_words * args.figure_content_share) * args.length_scale)
    target = max(600, target)
    log(f"source prose: {doc.prose_words} words -> target: {target} words")

    if args.dry_run:
        client: LLMClient = DryRunClient(doc)
    else:
        client = LLMClient(
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            retries=args.retries,
            cache_dir=args.cache_dir,
            debug_dir=args.debug_dir,
        )

    log("stage 1/5: outline")
    outline = build_outline(client, doc, target, args.max_source_chars)

    log("stage 2/5: figure captions")
    write_captions(client, doc, outline.title, args.workers,
                   args.keep_emphasis, audit=not args.no_desc_audit)

    log("stage 3/5: figure placement")
    place_figures(client, outline, doc, args.max_figs_per_section)

    log("stage 4/5: section writing")
    failures: List[str] = []
    lint_log: List[Tuple[str, List[Violation]]] = []
    sections = write_sections(client, outline, doc, args.workers,
                              args.keep_emphasis, args.max_tokens_section, failures,
                              linter, style, args.max_revisions, lint_log)

    log("stage 5/5: summary")
    summary = ""
    if sections and not args.no_summary:
        try:
            summary = clean_inline_md(client.chat(
                SYS_EDUCATOR, prompt_summary(outline, sections, style),
                max_tokens=MAXTOK_SUMMARY, temperature=TEMP_SUMMARY, tag="summary",
            ), args.keep_emphasis)
        except Exception as exc:                          # noqa: BLE001
            log(f"summary failed: {exc}")

    rendered_ids = {f.fid for s in sections for f in s.figures}
    orphans = [f for f in doc.figures if f.fid not in rendered_ids]

    output = render_document(outline, sections, doc, summary, args.model, orphans)

    problems = verify(output, doc)
    for f in failures:
        problems.append(f"section produced no prose: {f}")

    all_v = [v for _, vs in lint_log for v in vs]
    if all_v:
        log(f"residual style violations: {summarize_violations(all_v)}")
    orphan_facts = _unused_facts(doc, sections)
    if orphan_facts:
        problems.append(
            f"{len(orphan_facts)} claim(s) taken from figure descriptions do not "
            f"appear in any section's prose")
    if args.lint_report:
        _write_lint_report(args.lint_report, lint_log, orphan_facts)
        log(f"lint report written to {args.lint_report}")
    for p in problems:
        sys.stderr.write("warning: " + p + "\n")
    if failures and not args.debug_dir:
        sys.stderr.write(
            "hint: rerun with --debug-dir DIR to capture the raw completions for "
            "the failed sections, and consider --max-tokens-section 8192\n")

    if args.output and args.output != "-":
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(output)
        log(f"wrote {args.output}")
        print(f"{args.output}: {word_count(output)} words, "
              f"{len(doc.figures)} figures preserved, "
              f"{len(problems)} warning(s)")
    else:
        sys.stdout.write(output)

    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rewrite slide-derived medical markdown into fresh tagged material.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"backend: {DEFAULT_BASE_URL}  model: {DEFAULT_MODEL} ({MODEL_REPO})",
    )
    p.add_argument("input", nargs="?", help="input markdown file")
    p.add_argument("-o", "--output", default="-", help="output markdown file ('-' for stdout)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help="parallel LLM calls for captions and sections")
    p.add_argument("--target-words", type=int, default=0,
                   help="override the computed target length")
    p.add_argument("--length-scale", type=float, default=1.0,
                   help="multiplier on the source prose word count (default 1.0)")
    p.add_argument("--max-source-chars", type=int, default=14000,
                   help="how much source text to show the outline stage")
    p.add_argument("--cache-dir", default=None,
                   help="cache LLM responses here so reruns are cheap")
    p.add_argument("--debug-dir", default=None,
                   help="write every raw prompt+completion here for diagnosis")
    p.add_argument("--max-figs-per-section", type=int, default=MAX_FIGS_PER_SECTION,
                   help="overflow figures are moved to their next-best section "
                        "(0 disables rebalancing)")
    p.add_argument("--max-tokens-section", type=int, default=0,
                   help="override the per-section token budget (default: scaled "
                        "from the section's target length)")
    p.add_argument("--keep-emphasis", action="store_true",
                   help="keep inline **bold** / *italic* in generated text")
    p.add_argument("--max-revisions", type=int, default=1,
                   help="style-repair passes per section when the linter finds "
                        "violations (0 disables revision)")
    p.add_argument("--no-desc-audit", action="store_true",
                   help="trust the model's <desc>/<con> split without the "
                        "deterministic visual-sentence check")
    p.add_argument("--vague-term", action="append", default=[], metavar="TERM",
                   help="umbrella term that must not stand in for a specific "
                        "molecule, e.g. --vague-term 'thyroid hormone' "
                        "(repeatable)")
    p.add_argument("--figure-content-share", type=float, default=0.35,
                   help="fraction of figure-description words expected to migrate "
                        "into the prose, used to size the length target")
    p.add_argument("--lint-report", default=None,
                   help="write a style violation report to this path")
    p.add_argument("--no-summary", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="run the whole pipeline offline with stub completions")
    p.add_argument("--print-schema", action="store_true",
                   help="print the output tag vocabulary and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
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
