"""
Improved keywords generator for research queries.
"""

import logging
import re
from typing import Dict, List, Optional, Union

from academate.research_module.llm_response_handler import LLMResponseHandler, detect_llm_type


class ImprovedKeywordsGenerator:
    """
    Improved generator for research keywords.

    This class handles the generation of structured keywords from a research topic,
    organizing them into main keywords and synonyms for use in database queries.
    """

    def __init__(self, llm=None):
        """
        Initialize the KeywordsGenerator.

        Args:
            llm: An instance of the LLM for generating keywords
        """
        self.llm = llm

        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Set up response handler
        self.response_handler = LLMResponseHandler(logger=self.logger)

        # Detect LLM type for prompt formatting
        self.llm_type = detect_llm_type(llm)

    def generate_keywords(self, topic: str) -> str:
        """
        Generate structured keywords for the given topic.

        Args:
            topic: Research topic

        Returns:
            str: Structured keywords

        Raises:
            ValueError: If no LLM is provided
        """
        if not self.llm:
            raise ValueError("An LLM instance is required to generate keywords.")

        try:
            self.logger.info(f"Generating keywords for topic: {topic}")
            return self._generate_keyword_query(topic)
        except Exception as e:
            self.logger.error(f"Error generating keywords: {str(e)}")
            return self._create_fallback_keywords(topic)

    def _generate_keyword_query(self, topic: str) -> str:
        """
        Generate structured keywords using the LLM.

        Args:
            topic: Research topic

        Returns:
            str: Structured keywords
        """
        # Create prompt template
        prompt_template = f"""You are an expert in creating keywords for search queries in public databases like PubMed, Semantic Scholar, or Scopus.

Generate a comprehensive set of keywords describing the following research topic to improve search results in these databases:

Topic:
{topic}

**Instructions:**
1. Identify the main concepts in the topic.
2. For each concept, include synonyms and related terms.
3. Use the exact template below to format your response. Do not deviate from this structure.
4. Limit yourself to 3-5 main keywords.
5. Include 3-5 synonyms for each main keyword.
6. Do not include any additional information or context.

**Example of a correct response:**
<START_OUTPUT>
Keyword 1: [Main Keyword]
Synonyms: [Synonym 1], [Synonym 2], [Synonym 3]

Keyword 2: [Main Keyword]
Synonyms: [Synonym 1], [Synonym 2], [Synonym 3]
<END_OUTPUT>

**Please provide your response below:**
"""
        # Format prompt based on LLM type
        formatted_prompt = self.response_handler.format_prompt(prompt_template, self.llm_type)

        # Generate response
        response = self.llm.invoke(formatted_prompt)

        # Extract content from response
        content = self.response_handler.extract_content(response)

        # Extract keywords from content
        keywords = self._extract_keywords(content)

        return keywords

    def _extract_keywords(self, content: str) -> str:
        """
        Extract keywords from LLM response.

        Args:
            content: LLM response content

        Returns:
            str: Extracted keywords
        """
        # Look for content between markers
        start_marker = "<START_OUTPUT>"
        end_marker = "<END_OUTPUT>"

        if start_marker in content and end_marker in content:
            keywords_text = content.split(start_marker)[1].split(end_marker)[0].strip()
            self.logger.info("Successfully extracted keywords using markers")
            return keywords_text

        # If markers not found, try to extract using pattern matching
        self.logger.warning("Markers not found in response. Attempting pattern matching.")

        # Look for keyword pattern: "Keyword N: XXX"
        keyword_pattern = r'keyword\s+\d+\s*:\s*(.+?)(?=keyword\s+\d+\s*:|$)'
        keyword_matches = re.findall(keyword_pattern, content.lower(), re.DOTALL | re.IGNORECASE)

        if keyword_matches:
            self.logger.info("Extracted keywords using pattern matching")
            return content  # Return the whole content as patterns were found

        # If still no luck, use line-by-line extraction
        lines = content.strip().split('\n')
        found_keywords = []

        for line in lines:
            line = line.strip()
            # Look for lines starting with "Keyword" or containing ": " divider
            if line.lower().startswith("keyword") or ": " in line:
                found_keywords.append(line)

        if found_keywords:
            self.logger.info("Extracted keywords using line-by-line extraction")
            return "\n".join(found_keywords)

        # Last resort: return the whole content
        self.logger.warning("Could not extract structured keywords. Using full response.")
        return content

    def _create_fallback_keywords(self, topic: str) -> str:
        """
        Create a fallback keywords structure if generation fails.

        Args:
            topic: Research topic

        Returns:
            str: Basic keywords structure
        """
        # Split the topic into main terms
        terms = [t.strip() for t in re.split(r'[,;]|\band\b|\bor\b', topic) if t.strip()]

        # Create a basic keywords structure
        keywords = []

        for i, term in enumerate(terms[:5], 1):  # Limit to 5 main keywords
            synonyms = self._generate_simple_synonyms(term)
            keywords.append(f"Keyword {i}: {term}")
            keywords.append(f"Synonyms: {', '.join(synonyms)}")
            keywords.append("")  # Empty line

        return "\n".join(keywords).strip()

    def _generate_simple_synonyms(self, term: str) -> List[str]:
        """
        Generate simple synonyms for a term.

        Args:
            term: Main term

        Returns:
            List[str]: List of synonyms
        """
        # Create simple variations
        term = term.strip()
        words = term.split()

        if len(words) == 1:
            # For single words, add suffixes
            return [
                term,
                term + "s" if not term.endswith("s") else term,
                term + "ing" if not term.endswith("ing") else term,
                term + "ed" if not term.endswith("ed") else term
            ]
        else:
            # For phrases, rearrange or simplify
            return [
                term,
                " ".join(words[::-1]) if len(words) == 2 else term,  # Reverse for 2-word terms
                words[0] if words else term,  # First word
                words[-1] if words else term  # Last word
            ]

    def parse_keywords_to_dict(self, keywords_text: str) -> Dict[str, List[str]]:
        """
        Parse keywords text into a structured dictionary.

        Args:
            keywords_text: Generated keywords text

        Returns:
            Dict[str, List[str]]: Dictionary of main keywords and their synonyms
        """
        keyword_dict = {}
        current_keyword = None

        lines = [line.strip() for line in keywords_text.split('\n') if line.strip()]

        for line in lines:
            # Check for keyword line
            keyword_match = re.match(r'^keyword\s+\d+\s*:\s*(.+)$', line, re.IGNORECASE)
            if keyword_match:
                current_keyword = keyword_match.group(1).strip()
                keyword_dict[current_keyword] = []
                continue

            # Check for synonyms line
            synonyms_match = re.match(r'^synonyms\s*:\s*(.+)$', line, re.IGNORECASE)
            if synonyms_match and current_keyword:
                synonyms_text = synonyms_match.group(1)
                # Split by comma and strip whitespace
                synonyms = [s.strip() for s in re.split(r',\s*', synonyms_text) if s.strip()]
                keyword_dict[current_keyword] = synonyms

        return keyword_dict

    def format_for_database(self, keywords_text: str, database: str = "pubmed") -> str:
        """
        Format keywords for a specific database's syntax.

        Args:
            keywords_text: Generated keywords text
            database: Target database ("pubmed", "scopus", "semantic_scholar")

        Returns:
            str: Formatted query string for the specified database
        """
        # Parse keywords to dictionary
        keyword_dict = self.parse_keywords_to_dict(keywords_text)

        if not keyword_dict:
            return ""

        # Format based on database
        database = database.lower()

        if database == "pubmed":
            return self._format_for_pubmed(keyword_dict)
        elif database == "scopus":
            return self._format_for_scopus(keyword_dict)
        elif database == "semantic_scholar":
            return self._format_for_semantic_scholar(keyword_dict)
        else:
            self.logger.warning(f"Unknown database: {database}. Using generic format.")
            return self._format_generic(keyword_dict)

    def _format_for_pubmed(self, keyword_dict: Dict[str, List[str]]) -> str:
        """Format keywords for PubMed."""
        terms = []

        for main_keyword, synonyms in keyword_dict.items():
            # Combine main keyword with synonyms
            all_terms = [main_keyword] + synonyms
            # Wrap each term in quotes and join with OR
            term_group = " OR ".join([f'"{term}"[Title/Abstract]' for term in all_terms])
            # Wrap the group in parentheses
            terms.append(f"({term_group})")

        # Join all term groups with AND
        return " AND ".join(terms)

    def _format_for_scopus(self, keyword_dict: Dict[str, List[str]]) -> str:
        """Format keywords for Scopus."""
        terms = []

        for main_keyword, synonyms in keyword_dict.items():
            # Combine main keyword with synonyms
            all_terms = [main_keyword] + synonyms
            # Wrap each term in quotes and join with OR
            term_group = " OR ".join([f'"{term}"' for term in all_terms])
            # Wrap in TITLE-ABS-KEY
            terms.append(f"TITLE-ABS-KEY({term_group})")

        # Join all term groups with AND
        return " AND ".join(terms)

    def _format_for_semantic_scholar(self, keyword_dict: Dict[str, List[str]]) -> str:
        """Format keywords for Semantic Scholar."""
        terms = []

        for main_keyword, synonyms in keyword_dict.items():
            # Combine main keyword with synonyms
            all_terms = [main_keyword] + synonyms
            # Wrap each term in quotes and join with |
            term_group = " | ".join([f'"{term}"' for term in all_terms])
            # Wrap the group in parentheses
            terms.append(f"({term_group})")

        # Join all term groups with +
        return " + ".join(terms)

    def _format_generic(self, keyword_dict: Dict[str, List[str]]) -> str:
        """Format keywords generically."""
        lines = []

        for main_keyword, synonyms in keyword_dict.items():
            lines.append(f"Main term: {main_keyword}")
            lines.append(f"Synonyms: {', '.join(synonyms)}")
            lines.append("")

        return "\n".join(lines).strip()