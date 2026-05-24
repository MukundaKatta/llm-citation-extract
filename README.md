# llm-citation-extract

[![PyPI](https://img.shields.io/pypi/v/llm-citation-extract.svg)](https://pypi.org/project/llm-citation-extract/)
[![Python](https://img.shields.io/pypi/pyversions/llm-citation-extract.svg)](https://pypi.org/project/llm-citation-extract/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Extract citations and source references from LLM output.**

RAG-pipeline LLMs cite their sources in five or six different conventions
in the same response. This library normalizes them all into one typed
`Citation` list with stable span offsets, plus a `bibliography` map that
resolves numbered and footnote markers to their definitions.

Zero runtime deps. Stdlib `re` only.

## Install

```bash
pip install llm-citation-extract
```

## Use

```python
from llm_citation_extract import extract_citations

text = """
The capital is Paris [1]. Per Smith (2024), the population is ~2M.
See also [^foo] and https://example.com/source.
[1]: https://wikipedia.org/Paris
[^foo]: Smith, A. (2024). Demographics.
"""

result = extract_citations(text)
for c in result.citations:
    print(c.style, c.marker, "->", c.target)
```

Output:

```
CitationStyle.NUMBERED [1] -> https://wikipedia.org/Paris
CitationStyle.AUTHOR_YEAR (Smith, 2024) -> None
CitationStyle.FOOTNOTE [^foo] -> Smith, A. (2024). Demographics.
CitationStyle.URL https://example.com/source -> https://example.com/source
```

Each `Citation` is a frozen dataclass with `style`, `marker`, `anchor`,
`target`, and a `(start, end)` `span` into the original text.

## Supported styles

All six are extracted by default. Pass `styles=[...]` to filter.

### NUMBERED `[1]`, `[12]`

```python
text = "Paris is the capital [1].\n[1]: https://wikipedia.org/Paris\n"
extract_citations(text).citations[0].target
# 'https://wikipedia.org/Paris'
```

Resolves to a footer URL via `[N]: url` definition lines. Without a
matching definition, `target` is `None`.

### AUTHOR_YEAR `(Smith, 2024)`

Handles the common variants:

```python
extract_citations("Per (Smith, 2024), this works.")
extract_citations("Per (Smith and Jones, 2024), this works.")
extract_citations("Per (Smith et al., 2024), this works.")
```

Anchor is `"Smith 2024"` (whitespace collapsed), `target` is `None`.

### FOOTNOTE `[^foo]`

Pairs with a definition line:

```python
text = "Background [^foo].\n[^foo]: Smith, A. (2024). Demographics.\n"
extract_citations(text).citations[0].target
# 'Smith, A. (2024). Demographics.'
```

### URL bare http(s)

```python
extract_citations("See https://example.com/source for details.")
```

A URL inside a markdown link is not double-counted; the markdown link
wins.

### MARKDOWN_LINK `[text](url)`

```python
text = "Read [the post](https://example.com/post)."
c = extract_citations(text).citations[0]
c.anchor   # 'the post'
c.target   # 'https://example.com/post'
```

### PARENTHETICAL_ID `(SEC-001)`, `(RFC-2119)`

ALL-CAPS dashed tokens in parentheses.

```python
extract_citations("must obey (RFC-2119) per (SEC-001).")
```

A dash is required, so plain enclosures like `(NASA)` are skipped.

## Bibliography

Definition lines (`[1]: url` and `[^foo]: text`) are collected into a
single `bibliography` dict:

```python
result = extract_citations(text)
result.bibliography
# {'1': 'https://wikipedia.org/Paris', 'foo': 'Smith, A. (2024). Demographics.'}

result.bibliography_lookup("1")
# 'https://wikipedia.org/Paris'
```

## Clean text

The same call returns `clean_text` with inline markers and definition
lines stripped, so you can hand the prose to downstream summarizers or
embedders without citation noise:

```python
result = extract_citations(text)
print(result.clean_text)
# The capital is Paris . Per , the population is ~2M.
# See also  and .
```

## Style filter

Extract only the styles you care about:

```python
from llm_citation_extract import extract_citations, CitationStyle

extract_citations(text, styles=[CitationStyle.NUMBERED])
```

## What it does NOT do

- No fuzzy matching. A footnote marker `[^foo]` without a `[^foo]:`
  definition resolves to `target=None`. We never guess.
- No HTTP. We do not follow URLs to verify they exist.
- No formatter. We extract, we do not rewrite.
- No grammar parsing. The author-year regex needs a capitalized author
  token before the year; uncapitalized parentheticals like `(see, 2024)`
  are ignored on purpose.

## Siblings

Part of an LLM-output toolkit:

- [`driftvane`](https://pypi.org/project/driftvane/) - composable RAG
  and agent drift detectors (embedding, retrieval, response, latency).
- [`llm-output-validator`](https://pypi.org/project/llm-output-validator/) -
  rule-based validation of LLM outputs (regex, JSON shape, required
  keys).

## License

MIT
