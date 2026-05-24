"""Core citation extraction implementation.

RAG-pipeline LLMs cite sources in many different conventions. This module
normalizes them all into a `Citation` list with structured spans.

Supported styles:
  * NUMBERED         - `[1]`, `[12]` ... resolved via `[1]: url` definition lines
  * AUTHOR_YEAR      - `(Smith, 2024)`, `(Smith and Jones, 2024)`, `(Smith et al., 2024)`
  * FOOTNOTE         - `[^foo]` with `[^foo]: definition` line
  * URL              - bare http(s) URL outside of any other pattern
  * MARKDOWN_LINK    - `[text](url)`
  * PARENTHETICAL_ID - `(SEC-001)`, `(RFC-2119)`, ALL-CAPS dashed token
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class CitationStyle(str, Enum):
    """Citation style enum. Inherits from `str` so values are usable as keys
    and serializable as plain strings."""

    NUMBERED = "numbered"
    AUTHOR_YEAR = "author_year"
    FOOTNOTE = "footnote"
    URL = "url"
    MARKDOWN_LINK = "markdown_link"
    PARENTHETICAL_ID = "parenthetical_id"


_ALL_STYLES: frozenset[CitationStyle] = frozenset(CitationStyle)


@dataclass(frozen=True)
class Citation:
    """A single extracted citation occurrence.

    Attributes:
        style: which extraction style matched
        marker: the exact substring that matched in the source text
        anchor: normalized lookup key (e.g. "1", "foo", "Smith 2024"); None
            for URL-only matches that have no separate anchor
        target: resolved destination if extractable (URL for NUMBERED with a
            footer definition, definition text for FOOTNOTE, the URL itself
            for URL/MARKDOWN_LINK); None when nothing was resolvable
        span: (start, end) byte offsets into the original `text`
    """

    style: CitationStyle
    marker: str
    anchor: str | None
    target: str | None
    span: tuple[int, int]


@dataclass
class ExtractResult:
    """Result of `extract_citations()`.

    Attributes:
        citations: extracted citations in document order; duplicates preserved
        clean_text: original text with citation markers stripped; reference
            footers and footnote definitions are also dropped
        bibliography: anchor -> resolved target mapping built from footer
            definitions (`[1]: url`) and footnote definitions (`[^foo]: ...`)
    """

    citations: list[Citation] = field(default_factory=list)
    clean_text: str = ""
    bibliography: dict[str, str] = field(default_factory=dict)

    def bibliography_lookup(self, anchor: str) -> str | None:
        """Convenience: return the bibliography target for an anchor or None."""
        return self.bibliography.get(anchor)


# ---------- regex catalogue ----------

# A footer-style numbered reference: `[1]: https://...` or `[12]: text`.
# Anchored to a line start so it does not match inline `[1]` occurrences.
_NUMBERED_DEF_RE = re.compile(
    r"^[ \t]*\[(\d+)\]:[ \t]+(.+?)[ \t]*$",
    re.MULTILINE,
)

# Footnote definitions look like `[^foo]: text...`. Anchored to line start.
_FOOTNOTE_DEF_RE = re.compile(
    r"^[ \t]*\[\^([A-Za-z0-9_\-]+)\]:[ \t]+(.+?)[ \t]*$",
    re.MULTILINE,
)

# Inline numbered citation: `[1]`, `[12]`. Does not match definition lines
# (which have `]:` after the number); we exclude that by negative lookahead.
_NUMBERED_INLINE_RE = re.compile(r"\[(\d+)\](?!:)")

# Inline footnote reference: `[^foo]`. Excludes definition lines via lookahead.
_FOOTNOTE_INLINE_RE = re.compile(r"\[\^([A-Za-z0-9_\-]+)\](?!:)")

# Author-year citation. Handles two equivalent surface forms:
#
#   1. Author(s) INSIDE the parens:   (Smith, 2024)
#                                     (Smith and Jones, 2024)
#                                     (Smith et al., 2024)
#   2. Author(s) BEFORE the parens, with only the year inside:
#                                     Smith (2024)
#                                     Smith and Jones (2024)
#                                     Smith et al. (2024)
#
# We require at least one capitalized word for the author so we do not eat
# generic parenthetical text like `(see, 2024)` or `it (2024)` - capital-
# ization is the heuristic that separates author tokens from prose.
_AUTHOR_YEAR_INSIDE_RE = re.compile(
    r"\("
    r"(?P<authors>[A-Z][A-Za-z\-]+(?:\s+(?:and|et\sal\.|&)\s+[A-Z][A-Za-z\-]+|\s+et\sal\.)?)"
    r",\s*"
    r"(?P<year>\d{4})"
    r"\)"
)
_AUTHOR_YEAR_LEADING_RE = re.compile(
    r"(?P<authors>[A-Z][A-Za-z\-]+(?:\s+(?:and|et\sal\.|&)\s+[A-Z][A-Za-z\-]+|\s+et\sal\.)?)"
    r"\s+"
    r"\((?P<year>\d{4})\)"
)

# Markdown link: `[text](url)`. We capture the text and url separately.
# url must be non-empty and contain no whitespace or unbalanced parens.
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")

# Bare URL. Conservative: http/https, no trailing punctuation that is
# almost always sentence punctuation rather than part of the URL.
_URL_RE = re.compile(r"https?://[^\s<>\[\]()]+[^\s<>\[\]().,;!?]")

# Parenthetical ALL-CAPS dashed token: `(SEC-001)`, `(RFC-2119)`. Requires
# at least one dash so we do not match plain `(NASA)` or `(AI)` enclosures.
_PAREN_ID_RE = re.compile(r"\(([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)\)")


def _normalize_author_year(authors: str, year: str) -> str:
    """Build a stable anchor key from an author-year cite. Whitespace is
    collapsed so `(Smith,  2024)` and `(Smith, 2024)` share an anchor."""
    cleaned = re.sub(r"\s+", " ", authors).strip()
    return f"{cleaned} {year}"


def _build_bibliography(
    text: str,
) -> tuple[dict[str, str], dict[str, str], list[tuple[int, int]]]:
    """Find footer + footnote definitions.

    Returns (numbered_defs, footnote_defs, def_spans). `def_spans` is the
    list of (start, end) ranges to drop from `clean_text`.
    """
    numbered_defs: dict[str, str] = {}
    footnote_defs: dict[str, str] = {}
    def_spans: list[tuple[int, int]] = []

    for match in _NUMBERED_DEF_RE.finditer(text):
        anchor = match.group(1)
        target = match.group(2).strip()
        # First definition wins on duplicates so the bibliography stays stable.
        numbered_defs.setdefault(anchor, target)
        def_spans.append(match.span())

    for match in _FOOTNOTE_DEF_RE.finditer(text):
        anchor = match.group(1)
        target = match.group(2).strip()
        footnote_defs.setdefault(anchor, target)
        def_spans.append(match.span())

    return numbered_defs, footnote_defs, def_spans


def _strip_ranges(text: str, ranges: list[tuple[int, int]]) -> str:
    """Remove the given (start, end) byte ranges from `text`. Ranges may
    overlap; we merge them before stripping. Also collapses redundant blank
    lines that result from removing whole definition lines."""
    if not ranges:
        return text
    ordered = sorted(ranges)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    out = "".join(parts)
    # collapse stretches of 3+ newlines to 2, and trim leftover trailing
    # whitespace caused by stripping markers mid-line.
    out = re.sub(r"[ \t]+(\n)", r"\1", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def extract_citations(
    text: str,
    styles: list[CitationStyle] | set[CitationStyle] | frozenset[CitationStyle] | None = None,
) -> ExtractResult:
    """Extract citations from `text`.

    Args:
        text: source text from the LLM.
        styles: optional whitelist of styles to extract. Defaults to all six.

    Returns:
        `ExtractResult` with `citations` in document order, `clean_text`
        with markers and definition lines stripped, and `bibliography`
        mapping anchors to their resolved targets.
    """
    if styles is None:
        wanted: frozenset[CitationStyle] = _ALL_STYLES
    else:
        wanted = frozenset(styles)

    if not text:
        return ExtractResult(citations=[], clean_text="", bibliography={})

    numbered_defs, footnote_defs, def_spans = _build_bibliography(text)

    citations: list[Citation] = []
    # markers we want to strip from clean_text - includes definition lines
    # and all inline markers we extracted.
    strip_ranges: list[tuple[int, int]] = list(def_spans)
    # Track which (start, end) inline matches we have already taken so a
    # broader pattern (URL) does not double-count something already claimed
    # by a narrower pattern (MARKDOWN_LINK). We also seed `claimed` with the
    # definition-line spans so a URL appearing inside `[1]: https://...`
    # does not surface as a separate URL citation - it is part of the
    # bibliography target, not a body citation.
    claimed: list[tuple[int, int]] = list(def_spans)

    def _overlaps(span: tuple[int, int]) -> bool:
        s, e = span
        return any(not (e <= cs or s >= ce) for cs, ce in claimed)

    # ---- MARKDOWN_LINK first because URL would otherwise eat the url half ----
    if CitationStyle.MARKDOWN_LINK in wanted:
        for match in _MARKDOWN_LINK_RE.finditer(text):
            span = match.span()
            if _overlaps(span):
                continue
            link_text = match.group(1)
            url = match.group(2)
            citations.append(
                Citation(
                    style=CitationStyle.MARKDOWN_LINK,
                    marker=match.group(0),
                    anchor=link_text,
                    target=url,
                    span=span,
                )
            )
            claimed.append(span)
            strip_ranges.append(span)

    # ---- FOOTNOTE inline references ----
    if CitationStyle.FOOTNOTE in wanted:
        for match in _FOOTNOTE_INLINE_RE.finditer(text):
            span = match.span()
            if _overlaps(span):
                continue
            anchor = match.group(1)
            citations.append(
                Citation(
                    style=CitationStyle.FOOTNOTE,
                    marker=match.group(0),
                    anchor=anchor,
                    target=footnote_defs.get(anchor),
                    span=span,
                )
            )
            claimed.append(span)
            strip_ranges.append(span)

    # ---- NUMBERED inline references ----
    if CitationStyle.NUMBERED in wanted:
        for match in _NUMBERED_INLINE_RE.finditer(text):
            span = match.span()
            if _overlaps(span):
                continue
            anchor = match.group(1)
            citations.append(
                Citation(
                    style=CitationStyle.NUMBERED,
                    marker=match.group(0),
                    anchor=anchor,
                    target=numbered_defs.get(anchor),
                    span=span,
                )
            )
            claimed.append(span)
            strip_ranges.append(span)

    # ---- AUTHOR_YEAR ----
    # Two surface forms: `(Smith, 2024)` and `Smith (2024)`. Run both regex
    # passes and skip duplicates via the `claimed` overlap check.
    if CitationStyle.AUTHOR_YEAR in wanted:
        for regex in (_AUTHOR_YEAR_INSIDE_RE, _AUTHOR_YEAR_LEADING_RE):
            for match in regex.finditer(text):
                span = match.span()
                if _overlaps(span):
                    continue
                authors = match.group("authors")
                year = match.group("year")
                anchor = _normalize_author_year(authors, year)
                citations.append(
                    Citation(
                        style=CitationStyle.AUTHOR_YEAR,
                        marker=match.group(0),
                        anchor=anchor,
                        target=None,
                        span=span,
                    )
                )
                claimed.append(span)
                strip_ranges.append(span)

    # ---- PARENTHETICAL_ID ----
    if CitationStyle.PARENTHETICAL_ID in wanted:
        for match in _PAREN_ID_RE.finditer(text):
            span = match.span()
            if _overlaps(span):
                continue
            anchor = match.group(1)
            citations.append(
                Citation(
                    style=CitationStyle.PARENTHETICAL_ID,
                    marker=match.group(0),
                    anchor=anchor,
                    target=None,
                    span=span,
                )
            )
            claimed.append(span)
            strip_ranges.append(span)

    # ---- URL last so links inside other patterns are already claimed ----
    if CitationStyle.URL in wanted:
        for match in _URL_RE.finditer(text):
            span = match.span()
            if _overlaps(span):
                continue
            url = match.group(0)
            citations.append(
                Citation(
                    style=CitationStyle.URL,
                    marker=url,
                    anchor=None,
                    target=url,
                    span=span,
                )
            )
            claimed.append(span)
            strip_ranges.append(span)

    # preserve document order across all styles
    citations.sort(key=lambda c: c.span[0])

    bibliography: dict[str, str] = {}
    bibliography.update(numbered_defs)
    bibliography.update(footnote_defs)

    clean_text = _strip_ranges(text, strip_ranges).strip()

    return ExtractResult(
        citations=citations,
        clean_text=clean_text,
        bibliography=bibliography,
    )
