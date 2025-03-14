"""
Improved Semantic Scholar query generator.
"""

import pandas as pd
import logging
import requests
import time
from typing import Optional, Dict, List, Any, Union

from academate.research_module.query_extractor import SemanticScholarQueryExtractor
from academate.research_module.llm_response_handler import LLMResponseHandler, detect_llm_type

class ImprovedSemanticScholarQueryGenerator:
    """
    Improved generator for Semantic Scholar search queries.

    This class handles generation of Semantic Scholar queries using LLMs,
    with robust extraction and error handling.
    """

    def __init__(self, api_key: str, llm=None, two_step_search_query=False):
        """
        Initialize the SemanticScholarQueryGenerator.

        Args:
            api_key: Your Semantic Scholar API key
            llm: An instance of the LLM for generating queries
            two_step_search_query: Whether to use two-step approach
        """
        self.api_key = api_key
        self.llm = llm
        self.two_step_search_query = two_step_search_query

        # Set up headers for API calls
        self.headers = {"x-api-key": self.api_key}

        # Set up date filters
        self.from_year = None
        self.to_year = None

        # Set up results storage
        self.results = None
        self.last_query = ""

        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Set up response handler and query extractor
        self.response_handler = LLMResponseHandler(logger=self.logger)
        self.query_extractor = SemanticScholarQueryExtractor(logger=self.logger)

        # Detect LLM type for prompt formatting
        self.llm_type = detect_llm_type(llm)

    def set_date_filter(self, from_year: Optional[int] = None, to_year: Optional[int] = None) -> None:
        """
        Set date filter for the Semantic Scholar query.

        Args:
            from_year: Starting year (inclusive)
            to_year: Ending year (inclusive)
        """
        self.from_year = from_year
        self.to_year = to_year

    def generate_query(self, topic: str) -> str:
        """
        Generate a Semantic Scholar query for the given topic.

        Args:
            topic: Research topic or keywords

        Returns:
            str: Generated Semantic Scholar query

        Raises:
            ValueError: If no LLM is provided
        """
        if not self.llm:
            raise ValueError("An LLM instance is required to generate the query.")

        # Generate the query using the LLM
        max_attempts = 2
        for attempt in range(max_attempts):
            if self.two_step_search_query:
                query = self._generate_query_with_keywords(topic)
            else:
                query = self._generate_query_standard(topic)

            # Validate the query
            if self._validate_query(query):
                self.last_query = query
                return query
            else:
                self.logger.warning(f"Invalid query generated (attempt {attempt + 1}/{max_attempts}). Retrying...")

        # If all attempts failed, return a simplified query
        self.logger.warning("Could not generate a valid query. Using simplified query.")
        simplified_query = self._create_simplified_query(topic)
        self.last_query = simplified_query
        return simplified_query

    def _generate_query_standard(self, topic: str) -> str:
        """
        Generate a standard Semantic Scholar query using the LLM.

        Args:
            topic: Research topic

        Returns:
            str: Generated query
        """
        # Create prompt template
        prompt_template = f"""You are an expert in creating Semantic Scholar search queries.

Generate a comprehensive search query for the following research topic:

Topic:
{topic}

**Instructions:**
- Identify the main concepts in the topic.
- For each concept, include up to **three** synonyms or related terms.
- Use Boolean operators to combine terms:
    - Use **'+'** for **AND**.
    - Use **'|'** for **OR**.
    - Use **'-'** for **NOT**.
- Use quotation marks (**" "** ) for phrases.
- Use parentheses to group terms appropriately.
- Use wildcard (**'*'**) for word variations if appropriate.
- **Do not include any additional text, explanations, or notes.**
- **Provide ONLY the query, formatted correctly for the Semantic Scholar API, and nothing else.**
- **Enclose your query between the markers `<START_QUERY>` and `<END_QUERY>`. Failure to do so will result in an invalid query.**

**Example of a correct response:**

<START_QUERY>
("term1" | "term1 synonym" | "term1 related term") + ("term2" | "term2 synonym")
<END_QUERY>

Provide your query below:
"""
        # Format prompt based on LLM type
        formatted_prompt = self.response_handler.format_prompt(prompt_template, self.llm_type)

        # Generate response
        try:
            response = self.llm.invoke(formatted_prompt)

            # Extract content from response
            content = self.response_handler.extract_content(response)

            # Extract query from content
            query = self.query_extractor.extract_query(content)

            return query

        except Exception as e:
            self.logger.error(f"Error generating query: {str(e)}")
            # Return a basic query as fallback
            return f'("{topic}")'

    def _generate_query_with_keywords(self, keywords: str) -> str:
        """
        Generate a Semantic Scholar query using the LLM based on predefined keywords.

        Args:
            keywords: Keywords and terms for the query

        Returns:
            str: Generated query
        """
        # Create prompt template
        prompt_template = f"""You are an expert in creating Semantic Scholar search queries.

Generate a comprehensive search query for the following keywords:
{keywords}

**Instructions:**
- Use Boolean operators to combine terms:
    - Use **'+'** for **AND**.
    - Use **'|'** for **OR**.
    - Use **'-'** for **NOT**.
- Use quotation marks (**" "** ) for phrases.
- Use parentheses to group terms appropriately.
- **Do not include any additional text, explanations, or notes.**
- **Provide ONLY the query, formatted correctly for the Semantic Scholar API, and nothing else.**
- **Enclose your query between the markers `<START_QUERY>` and `<END_QUERY>`. Failure to do so will result in an invalid query.**

**Example of a correct response:**

<START_QUERY>
("term1" | "term1 synonym" | "term1 related term") + ("term2" | "term2 synonym")
<END_QUERY>

Provide your query below:
"""
        # Format prompt based on LLM type
        formatted_prompt = self.response_handler.format_prompt(prompt_template, self.llm_type)

        # Generate response
        try:
            response = self.llm.invoke(formatted_prompt)

            # Extract content from response
            content = self.response_handler.extract_content(response)

            # Extract query from content
            query = self.query_extractor.extract_query(content)

            return query

        except Exception as e:
            self.logger.error(f"Error generating query: {str(e)}")
            # Return a basic query constructed from keywords
            lines = [line.strip() for line in keywords.split('\n') if line.strip()]
            terms = []
            for line in lines:
                if ':' in line:
                    term = line.split(':', 1)[1].strip()
                    if term:
                        terms.append(term)

            if terms:
                return ' + '.join([f'("{term}")' for term in terms])
            else:
                return f'("{keywords}")'

    def _validate_query(self, query: str) -> bool:
        """
        Validate the query for correct syntax.

        Args:
            query: Query string to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if not query or len(query) < 3:
            return False

        # Check for balanced parentheses
        if query.count('(') != query.count(')'):
            self.logger.error("Unbalanced parentheses in query")
            return False

        # Check for unmatched quotes
        if query.count('"') % 2 != 0:
            self.logger.error("Unmatched quotes in query")
            return False

        # Check for invalid operators
        if '{' in query or '}' in query:
            self.logger.error("Invalid brackets in query")
            return False

        # Ensure query contains at least one term
        if '"' not in query and "'" not in query:
            self.logger.warning("Query does not contain any quoted terms")
            return False

        return True

    def _create_simplified_query(self, topic: str) -> str:
        """
        Create a simplified query for fallback purposes.

        Args:
            topic: Research topic

        Returns:
            str: Simplified query
        """
        # Extract key terms from the topic
        terms = [term.strip() for term in topic.split() if len(term.strip()) > 3]

        # Create a simplified query with the main terms
        if len(terms) > 3:
            main_terms = terms[:3]  # Use only the first 3 terms
            return ' + '.join([f'"{term}"' for term in main_terms])
        elif terms:
            return ' + '.join([f'"{term}"' for term in terms])
        else:
            return f'"{topic}"'

    def execute_query(self, query: str, max_results: int = 1000) -> None:
        """
        Execute the Semantic Scholar query and retrieve results.

        Args:
            query: Semantic Scholar query string
            max_results: Maximum number of results to retrieve
        """
        try:
            self.logger.info(f"Executing Semantic Scholar query: {query}")

            params = {
                "query": query,
                "fields": "paperId,title,abstract,authors,year,venue,externalIds",
                "limit": 100  # Maximum allowed per request
            }

            # Apply date filters
            if self.from_year or self.to_year:
                publication_date = ""
                if self.from_year:
                    publication_date += f"{self.from_year}"
                publication_date += ":"
                if self.to_year:
                    publication_date += f"{self.to_year}"
                params['publicationDateOrYear'] = publication_date

            papers = []
            total_results = 0
            next_token = None

            while total_results < max_results:
                if next_token:
                    params['token'] = next_token

                # Add delay to avoid rate limiting
                time.sleep(1)

                response = requests.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search/bulk",
                    headers=self.headers,
                    params=params
                )

                if response.status_code == 200:
                    data = response.json()
                    batch_papers = data.get('data', [])
                    papers.extend(batch_papers)
                    total_results += len(batch_papers)

                    # Log progress
                    self.logger.info(f"Retrieved {len(batch_papers)} papers (total: {total_results})")

                    # Check for next token
                    if 'next' in data and data['next']:
                        next_token = data['next']
                    else:
                        break  # No more results
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    self.logger.warning("Rate limited. Waiting before retrying...")
                    time.sleep(10)
                    continue
                else:
                    self.logger.error(f"API error: {response.status_code} - {response.text}")
                    break

                # Break if we've reached max_results
                if total_results >= max_results:
                    break

            # Process papers into DataFrame
            self.results = self._process_papers(papers[:max_results])
            self.logger.info(f"Successfully retrieved {len(self.results)} papers")

        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            self.results = pd.DataFrame()

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get query results as a DataFrame.

        Returns:
            pd.DataFrame: Results DataFrame
        """
        return self.results

    def _process_papers(self, papers: List[Dict]) -> pd.DataFrame:
        """
        Process the list of papers and convert to DataFrame.

        Args:
            papers: List of paper dictionaries from API

        Returns:
            pd.DataFrame: Processed DataFrame
        """
        if not papers:
            return pd.DataFrame()

        try:
            titles = []
            abstracts = []
            authors_list = []
            years = []
            venues = []
            paper_ids = []
            dois = []

            for paper in papers:
                # Extract basic info
                titles.append(paper.get('title', ''))
                abstracts.append(paper.get('abstract', ''))

                # Extract authors
                authors = [author.get('name', '') for author in paper.get('authors', [])]
                authors_list.append(', '.join(authors))

                # Extract year and venue
                years.append(paper.get('year', ''))
                venues.append(paper.get('venue', ''))

                # Extract IDs
                paper_ids.append(paper.get('paperId', ''))
                external_ids = paper.get('externalIds', {})
                doi = external_ids.get('DOI', '')
                dois.append(doi)

            # Create DataFrame
            df = pd.DataFrame({
                "Paper ID": paper_ids,
                "Title": titles,
                "Abstract": abstracts,
                "Authors": authors_list,
                "Year": years,
                "Venue": venues,
                "DOI": dois
            })

            return df

        except Exception as e:
            self.logger.error(f"Error processing papers: {str(e)}")
            return pd.DataFrame()