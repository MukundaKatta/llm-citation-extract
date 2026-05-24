"""llm-citation-extract - normalize LLM citations into structured spans.

RAG-pipeline LLMs cite their sources in many different conventions:
numbered footnotes, author-year, markdown links, bare URLs, parenthetical
IDs. This library extracts them all into a single typed `Citation` list
with stable span offsets, plus a `bibliography` map that resolves
numbered and footnote markers to their definitions.

    from llm_citation_extract import extract_citations, CitationStyle

    text = '''
    The capital is Paris [1]. Per Smith (2024), the population is ~2M.
    See also [^foo] and https://example.com/source.
    [1]: https://wikipedia.org/Paris
    [^foo]: Smith, A. (2024). Demographics.
    '''

    result = extract_citations(text)
    for c in result.citations:
        print(c.style, c.marker, "->", c.target)

Filter by style:

    only_numbered = extract_citations(text, styles=[CitationStyle.NUMBERED])

Sibling to `driftvane` (RAG drift) and `llm-output-validator`
(rule-based output validation).
"""

from llm_citation_extract.extract import (
    Citation,
    CitationStyle,
    ExtractResult,
    extract_citations,
)

__version__ = "0.1.0"

__all__ = [
    "Citation",
    "CitationStyle",
    "ExtractResult",
    "__version__",
    "extract_citations",
]
