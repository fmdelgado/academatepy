"""
Query extraction utilities for literature search.
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union


class BaseQueryExtractor(ABC):
    """Base class for extracting queries from LLM outputs."""

    def __init__(self, logger=None):
        """Initialize with optional logger."""
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def extract_query(self, content: str) -> str:
        """Extract query from the LLM output content."""
        pass

    def _find_text_between_markers(self, content: str, start_marker: str, end_marker: str) -> Optional[str]:
        """Find text between specified markers."""
        pattern = f"{re.escape(start_marker)}(.*?){re.escape(end_marker)}"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _extract_from_code_block(self, content: str) -> Optional[str]:
        """Extract content from code blocks."""
        code_blocks = re.findall(r'```(?:\w+)?\s*(.*?)\s*```', content, re.DOTALL)
        return code_blocks[0].strip() if code_blocks else None

    def _first_paragraph(self, content: str) -> str:
        """Get the first non-empty paragraph or line."""
        paragraphs = re.split(r'\n\s*\n', content)
        for p in paragraphs:
            p = p.strip()
            if p:
                return p

        # If no paragraphs, try lines
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        return lines[0] if lines else content.strip()


class MarkedQueryExtractor(BaseQueryExtractor):
    """Extract queries using start and end markers."""

    def __init__(self, start_marker="<START_QUERY>", end_marker="<END_QUERY>", logger=None):
        """Initialize with specified markers."""
        super().__init__(logger)
        self.start_marker = start_marker
        self.end_marker = end_marker

    def extract_query(self, content: str) -> str:
        """Extract query between markers with fallbacks."""
        # Try to extract between markers first
        query = self._find_text_between_markers(content, self.start_marker, self.end_marker)

        if not query:
            self.logger.warning(f"Markers '{self.start_marker}' and '{self.end_marker}' not found. Trying fallbacks.")

            # Try code blocks
            query = self._extract_from_code_block(content)
            if query:
                self.logger.info("Extracted query from code block.")
                return query

            # Try JSON extraction if it looks like JSON
            if '{' in content and '}' in content:
                try:
                    import json
                    # Find all JSON-like patterns
                    json_matches = re.findall(r'\{.*\}', content, re.DOTALL)
                    for json_text in json_matches:
                        try:
                            data = json.loads(json_text)
                            if 'query' in data:
                                self.logger.info("Extracted query from JSON.")
                                return data['query']
                        except json.JSONDecodeError:
                            continue
                except Exception:
                    pass

            # Fall back to first paragraph/line
            query = self._first_paragraph(content)
            self.logger.warning("Using first paragraph as query (fallback method).")

        return query or ""


class PubMedQueryExtractor(MarkedQueryExtractor):
    """Specialized query extractor for PubMed."""

    def extract_query(self, content: str) -> str:
        """Extract and clean query for PubMed format."""
        query = super().extract_query(content)
        return self._clean_pubmed_query(query)

    def _clean_pubmed_query(self, query: str) -> str:
        """Clean and format query for PubMed."""
        if not query:
            return ""

        # Ensure proper parentheses
        if not query.startswith("("):
            query = "(" + query
        if not query.endswith(")"):
            query = query + ")"

        # Fix spacing around AND, OR, NOT operators
        query = re.sub(r'(\w+)\s+(AND|OR|NOT)\s+', r'\1 \2 ', query)

        return query


class ScopusQueryExtractor(MarkedQueryExtractor):
    """Specialized query extractor for Scopus."""

    def extract_query(self, content: str) -> str:
        """Extract and clean query for Scopus format."""
        query = super().extract_query(content)
        return self._clean_scopus_query(query)

    def _clean_scopus_query(self, query: str) -> str:
        """Clean and format query for Scopus."""
        if not query:
            return ""

        # Fix quotes
        query = query.replace('"', '"').replace('"', '"')

        # Ensure field codes are valid
        # This is a simplified check - you may want to expand this
        valid_field_codes = ['TITLE', 'ABS', 'KEY', 'AUTH', 'AFFIL', 'SRCTITLE',
                             'DOCTYPE', 'LANGUAGE', 'PUBYEAR']

        # Check for balanced parentheses
        if query.count('(') != query.count(')'):
            self.logger.warning("Unbalanced parentheses in Scopus query")
            # Simple fix for unbalanced parentheses
            while query.count('(') > query.count(')'):
                query += ')'
            while query.count('(') < query.count(')'):
                query = '(' + query

        return query


class SemanticScholarQueryExtractor(MarkedQueryExtractor):
    """Specialized query extractor for Semantic Scholar."""

    def extract_query(self, content: str) -> str:
        """Extract and clean query for Semantic Scholar format."""
        query = super().extract_query(content)
        return self._clean_semantic_scholar_query(query)

    def _clean_semantic_scholar_query(self, query: str) -> str:
        """Clean and format query for Semantic Scholar."""
        if not query:
            return ""

        # Replace curly braces with parentheses
        query = query.replace('{', '(').replace('}', ')')

        # Replace fancy quotes with straight quotes
        query = query.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")

        # Check for balanced parentheses
        if query.count('(') != query.count(')'):
            self.logger.warning("Unbalanced parentheses in Semantic Scholar query")
            # Simple fix for unbalanced parentheses
            while query.count('(') > query.count(')'):
                query += ')'
            while query.count('(') < query.count(')'):
                query = '(' + query

        return query


# Factory function to get the appropriate extractor
def get_query_extractor(database_type: str, logger=None) -> BaseQueryExtractor:
    """
    Get a query extractor for the specified database type.

    Args:
        database_type: Type of database ('pubmed', 'scopus', 'semantic_scholar', etc.)
        logger: Optional logger instance

    Returns:
        Appropriate query extractor instance
    """
    extractors = {
        'pubmed': PubMedQueryExtractor,
        'scopus': ScopusQueryExtractor,
        'semantic_scholar': SemanticScholarQueryExtractor,
        'embase': MarkedQueryExtractor,  # Use base class for Embase
        'default': MarkedQueryExtractor,
    }

    extractor_class = extractors.get(database_type.lower(), extractors['default'])
    return extractor_class(logger=logger)