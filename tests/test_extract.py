"""Tests for llm_citation_extract.extract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from llm_citation_extract import Citation, CitationStyle, ExtractResult, extract_citations

# ---------- empty / no-citation cases ----------


def test_empty_text_returns_empty_result():
    r = extract_citations("")
    assert r.citations == []
    assert r.clean_text == ""
    assert r.bibliography == {}


def test_text_with_no_citations_returns_empty_list():
    r = extract_citations("just a sentence with nothing special.")
    assert r.citations == []
    assert r.bibliography == {}
    assert "just a sentence" in r.clean_text


# ---------- NUMBERED ----------


def test_numbered_single_inline():
    r = extract_citations("Paris is the capital [1].")
    assert len(r.citations) == 1
    c = r.citations[0]
    assert c.style == CitationStyle.NUMBERED
    assert c.marker == "[1]"
    assert c.anchor == "1"
    assert c.target is None


def test_numbered_resolves_to_footer_url():
    text = "Paris is the capital [1].\n\n[1]: https://wikipedia.org/Paris\n"
    r = extract_citations(text)
    assert len(r.citations) == 1
    c = r.citations[0]
    assert c.style == CitationStyle.NUMBERED
    assert c.anchor == "1"
    assert c.target == "https://wikipedia.org/Paris"
    assert r.bibliography["1"] == "https://wikipedia.org/Paris"


def test_numbered_multi_digit():
    text = "see [12] and [3].\n[3]: a\n[12]: b\n"
    r = extract_citations(text)
    anchors = [c.anchor for c in r.citations if c.style == CitationStyle.NUMBERED]
    assert anchors == ["12", "3"]
    assert r.bibliography == {"3": "a", "12": "b"}


# ---------- AUTHOR_YEAR ----------


def test_author_year_single_paren_form():
    # author and year both inside the parens
    r = extract_citations("Per (Smith, 2024), this works.")
    cites = [c for c in r.citations if c.style == CitationStyle.AUTHOR_YEAR]
    assert len(cites) == 1
    assert cites[0].marker == "(Smith, 2024)"
    assert cites[0].anchor == "Smith 2024"
    assert cites[0].target is None


def test_author_year_leading_form():
    # author outside parens, only year inside (`Smith (2024)`)
    r = extract_citations("Per Smith (2024), this works.")
    cites = [c for c in r.citations if c.style == CitationStyle.AUTHOR_YEAR]
    assert len(cites) == 1
    assert cites[0].marker == "Smith (2024)"
    assert cites[0].anchor == "Smith 2024"


def test_author_year_two_authors():
    r = extract_citations("Per (Smith and Jones, 2024), this works.")
    cites = [c for c in r.citations if c.style == CitationStyle.AUTHOR_YEAR]
    assert len(cites) == 1
    assert cites[0].marker == "(Smith and Jones, 2024)"
    assert cites[0].anchor == "Smith and Jones 2024"


def test_author_year_et_al():
    r = extract_citations("Per (Smith et al., 2024), this works.")
    cites = [c for c in r.citations if c.style == CitationStyle.AUTHOR_YEAR]
    assert len(cites) == 1
    assert "et al." in cites[0].marker
    assert cites[0].anchor == "Smith et al. 2024"


def test_author_year_ampersand_separator():
    # `&` is an accepted author separator alongside `and`.
    r = extract_citations("Per (Smith & Jones, 2024), this works.")
    cites = [c for c in r.citations if c.style == CitationStyle.AUTHOR_YEAR]
    assert len(cites) == 1
    assert cites[0].marker == "(Smith & Jones, 2024)"
    assert cites[0].anchor == "Smith & Jones 2024"


def test_author_year_ignores_generic_parens():
    r = extract_citations("It (see, 2024) does not match.")
    cites = [c for c in r.citations if c.style == CitationStyle.AUTHOR_YEAR]
    assert cites == []


# ---------- FOOTNOTE ----------


def test_footnote_resolves_to_definition():
    text = "Background [^foo].\n\n[^foo]: Smith, A. (2024). Demographics.\n"
    r = extract_citations(text)
    foot = [c for c in r.citations if c.style == CitationStyle.FOOTNOTE]
    assert len(foot) == 1
    assert foot[0].anchor == "foo"
    assert foot[0].target == "Smith, A. (2024). Demographics."
    assert r.bibliography["foo"] == "Smith, A. (2024). Demographics."


def test_footnote_without_definition_has_no_target():
    r = extract_citations("Background [^lonely].")
    foot = [c for c in r.citations if c.style == CitationStyle.FOOTNOTE]
    assert len(foot) == 1
    assert foot[0].anchor == "lonely"
    assert foot[0].target is None


# ---------- URL ----------


def test_bare_url_extracted():
    r = extract_citations("See https://example.com/source for details.")
    urls = [c for c in r.citations if c.style == CitationStyle.URL]
    assert len(urls) == 1
    assert urls[0].marker == "https://example.com/source"
    assert urls[0].target == "https://example.com/source"
    assert urls[0].anchor is None


def test_url_does_not_consume_trailing_period():
    r = extract_citations("See https://example.com/source.")
    urls = [c for c in r.citations if c.style == CitationStyle.URL]
    assert len(urls) == 1
    assert urls[0].marker == "https://example.com/source"


def test_url_not_double_counted_inside_markdown_link():
    text = "Read [the post](https://example.com/post)."
    r = extract_citations(text)
    styles = [c.style for c in r.citations]
    assert CitationStyle.MARKDOWN_LINK in styles
    assert CitationStyle.URL not in styles


# ---------- MARKDOWN_LINK ----------


def test_markdown_link_target_is_url():
    text = "Read [the post](https://example.com/post)."
    r = extract_citations(text)
    md = [c for c in r.citations if c.style == CitationStyle.MARKDOWN_LINK]
    assert len(md) == 1
    assert md[0].anchor == "the post"
    assert md[0].target == "https://example.com/post"


# ---------- PARENTHETICAL_ID ----------


def test_parenthetical_id_matches_caps_dashed():
    r = extract_citations("must obey (RFC-2119) per (SEC-001).")
    pid = [c for c in r.citations if c.style == CitationStyle.PARENTHETICAL_ID]
    assert [c.anchor for c in pid] == ["RFC-2119", "SEC-001"]


def test_parenthetical_id_requires_dash():
    # plain `(NASA)` is not a parenthetical id - no dash
    r = extract_citations("the agency (NASA) confirmed this.")
    pid = [c for c in r.citations if c.style == CitationStyle.PARENTHETICAL_ID]
    assert pid == []


# ---------- order, dedupe, mixed ----------


def test_order_preserved_in_document_order():
    text = (
        "First [1], then (Smith, 2024), then [^foo], then "
        "https://example.com and finally (RFC-2119)."
    )
    r = extract_citations(text)
    styles_in_order = [c.style for c in r.citations]
    assert styles_in_order == [
        CitationStyle.NUMBERED,
        CitationStyle.AUTHOR_YEAR,
        CitationStyle.FOOTNOTE,
        CitationStyle.URL,
        CitationStyle.PARENTHETICAL_ID,
    ]


def test_dedupe_keeps_each_appearance():
    text = "First [1] and again [1] then [1].\n[1]: https://x\n"
    r = extract_citations(text)
    numbered = [c for c in r.citations if c.style == CitationStyle.NUMBERED]
    assert len(numbered) == 3
    assert all(c.anchor == "1" for c in numbered)
    assert all(c.target == "https://x" for c in numbered)


def test_citation_at_end_of_sentence():
    r = extract_citations("Confirmed [1].")
    nums = [c for c in r.citations if c.style == CitationStyle.NUMBERED]
    assert len(nums) == 1
    start, end = nums[0].span
    assert r.citations[0].marker == "[1]"
    # span points exactly at "[1]"
    assert "Confirmed [1]."[start:end] == "[1]"


def test_citation_in_title_position():
    # leading-line cite should still be captured
    text = "[1] is the only source.\n[1]: https://x\n"
    r = extract_citations(text)
    assert any(c.style == CitationStyle.NUMBERED for c in r.citations)


# ---------- bibliography ----------


def test_bibliography_combines_numbered_and_footnote():
    text = "see [1] and [^foo]\n[1]: https://x\n[^foo]: a definition line\n"
    r = extract_citations(text)
    assert r.bibliography == {
        "1": "https://x",
        "foo": "a definition line",
    }
    assert r.bibliography_lookup("1") == "https://x"
    assert r.bibliography_lookup("foo") == "a definition line"
    assert r.bibliography_lookup("missing") is None


# ---------- clean_text ----------


def test_clean_text_strips_inline_markers_and_definitions():
    text = "Paris is the capital [1].\n\n[1]: https://wikipedia.org/Paris\n"
    r = extract_citations(text)
    # marker stripped
    assert "[1]" not in r.clean_text
    # definition line stripped
    assert "wikipedia.org" not in r.clean_text
    # core sentence preserved
    assert "Paris is the capital" in r.clean_text


def test_clean_text_strips_markdown_link():
    text = "Read [the post](https://example.com/post) now."
    r = extract_citations(text)
    assert "[the post]" not in r.clean_text
    assert "https://example.com" not in r.clean_text
    assert "Read" in r.clean_text
    assert "now." in r.clean_text


# ---------- style filter ----------


def test_style_filter_only_numbered():
    text = "see [1] and (Smith, 2024) plus https://x"
    r = extract_citations(text, styles=[CitationStyle.NUMBERED])
    assert len(r.citations) == 1
    assert r.citations[0].style == CitationStyle.NUMBERED


def test_style_filter_excludes_url_when_not_listed():
    text = "see https://example.com here"
    r = extract_citations(text, styles=[CitationStyle.NUMBERED, CitationStyle.AUTHOR_YEAR])
    assert r.citations == []


def test_style_filter_accepts_set():
    text = "see [1] and (Smith, 2024)"
    r = extract_citations(text, styles={CitationStyle.AUTHOR_YEAR})
    assert len(r.citations) == 1
    assert r.citations[0].style == CitationStyle.AUTHOR_YEAR


# ---------- whole-document smoke test ----------


def test_full_example_from_readme():
    text = (
        "The capital is Paris [1]. Per Smith (2024), the population is ~2M.\n"
        "See also [^foo] and https://example.com/source.\n"
        "[1]: https://wikipedia.org/Paris\n"
        "[^foo]: Smith, A. (2024). Demographics.\n"
    )
    r = extract_citations(text)
    assert isinstance(r, ExtractResult)
    styles = [c.style for c in r.citations]
    assert CitationStyle.NUMBERED in styles
    assert CitationStyle.AUTHOR_YEAR in styles
    assert CitationStyle.FOOTNOTE in styles
    assert CitationStyle.URL in styles
    # bibliography resolves both
    numbered = next(c for c in r.citations if c.style == CitationStyle.NUMBERED)
    foot = next(c for c in r.citations if c.style == CitationStyle.FOOTNOTE)
    assert numbered.target == "https://wikipedia.org/Paris"
    assert foot.target == "Smith, A. (2024). Demographics."
    # `Per Smith (2024)` is the leading author-year form, so the marker is
    # `Smith (2024)` (not `(Smith, 2024)`); guards the README output block.
    author_year = next(c for c in r.citations if c.style == CitationStyle.AUTHOR_YEAR)
    assert author_year.marker == "Smith (2024)"
    # the `(2024)` inside the `[^foo]:` definition line must not surface as a
    # separate author-year citation.
    assert sum(1 for c in r.citations if c.style == CitationStyle.AUTHOR_YEAR) == 1


# ---------- Citation type contract ----------


def test_citation_is_frozen():
    c = Citation(
        style=CitationStyle.NUMBERED,
        marker="[1]",
        anchor="1",
        target=None,
        span=(0, 3),
    )
    with pytest.raises(FrozenInstanceError):
        c.marker = "[2]"  # type: ignore[misc]


def test_citation_style_is_string_enum():
    # StrEnum-style: value is a str subclass so direct comparison works.
    assert CitationStyle.NUMBERED == "numbered"
    assert isinstance(CitationStyle.NUMBERED.value, str)
