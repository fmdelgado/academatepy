import pandas as pd
import re
import logging
import requests
from typing import Optional
import asyncio
import aiohttp
from functools import lru_cache


class ScopusQueryGenerator:
    def __init__(self, api_key: str, llm=None):
        """
        Initializes the ScopusQueryGenerator.

        Parameters:
        - api_key (str): Your Elsevier Scopus API key.
        - llm: An instance of the LLM for generating and improving queries.
        """
        self.api_key = api_key
        self.llm = llm
        self.from_year = None
        self.to_year = None
        self.results = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.base_url = "https://api.elsevier.com/content/search/scopus"

    def set_date_filter(self, from_year: Optional[int] = None, to_year: Optional[int] = None):
        """
        Sets the date filter for the Scopus query.
        """
        self.from_year = from_year
        self.to_year = to_year

    def get_date_filter(self) -> str:
        date_filters = []
        if self.from_year:
            date_filters.append(f"PUBYEAR AFT {self.from_year - 1}")
        if self.to_year:
            date_filters.append(f"PUBYEAR BEF {self.to_year + 1}")
        return ' AND '.join(date_filters)

    @lru_cache(maxsize=32)
    def generate_query(self, topic: str) -> str:
        """
        Generates a Scopus advanced search query for the given topic.
        """
        if not self.llm:
            raise ValueError("An LLM instance is required to generate the query.")

        query = self.generate_scopus_query(topic)

        # Apply date filters if set
        if self.from_year or self.to_year:
            date_filter = self.get_date_filter()
            query = f"({query}) AND {date_filter}"

        # Validate the query
        if not self.validate_query(query):
            raise ValueError("Generated query is invalid.")

        return query

    def generate_scopus_query(self, topic: str) -> str:
        """
        Generates a Scopus advanced search query using the LLM.
        """
        prompt = f"""You are an expert in creating Scopus advanced search queries.

    Generate a comprehensive Scopus search query for the following research topic:

    Topic:
    {topic}

    Requirements:
    1. Identify the main concepts in the topic.
    2. For each concept, include synonyms and related terms.
    3. Use valid Scopus field codes such as TITLE(), ABS(), KEY(), AUTH(), AFFIL(), SRCTITLE(), DOCTYPE(), LANGUAGE(), PUBYEAR().
    4. Do not use invalid field codes or unsupported syntax like LIMIT-TO(), SUBJECT(), SUBFIELD(), or AUTH-KEY().
    5. Use Boolean operators (AND, OR, AND NOT) to create effective query structures.
    6. Balance specificity and sensitivity.
    7. Do not include any additional text, explanations, or notes.
    8. Provide ONLY the query, formatted correctly for Scopus advanced search.
    9. Enclose your query between the markers <START_QUERY> and <END_QUERY>. Failure to do so will result in an invalid query.

    Example:

    <START_QUERY>
    (TITLE-ABS-KEY("term1" OR "term1 synonym") AND TITLE-ABS-KEY("term2" OR "term2 synonym"))
    <END_QUERY>

    Provide your query below:
    """

        response = self.llm.invoke(prompt)
        content = self.extract_content(response)
        query = self.extract_query_from_markers(content)
        query = self.clean_query(query)
        return query

    def extract_query_from_markers(self, content: str) -> str:
        """
        Extracts the query between the markers <START_QUERY> and <END_QUERY>.
        """
        match = re.search(r'<START_QUERY>(.*?)<END_QUERY>', content, re.DOTALL | re.IGNORECASE)
        if match:
            query = match.group(1).strip()
        else:
            self.logger.warning("Markers not found in LLM output. Using entire content as query.")
            query = content.strip()
        return query

    def clean_query(self, query: str) -> str:
        """
        Cleans and formats the generated query.
        """
        if not query:
            return ""

        # Remove leading/trailing whitespace and unwanted characters
        query = query.strip("`'\" \n\t")

        # Remove any accidental markers or extra text
        query = re.sub(r'<.*?>', '', query)

        # Replace fancy quotes with straight quotes
        query = query.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")

        # Ensure proper spacing around parentheses
        query = re.sub(r'\s*\(\s*', ' (', query)
        query = re.sub(r'\s*\)\s*', ') ', query)

        # Replace multiple spaces with a single space
        query = re.sub(r'\s+', ' ', query).strip()

        # Ensure balanced parentheses
        if query.count('(') != query.count(')'):
            self.logger.warning("Unbalanced parentheses in the query.")

        return query

    def validate_query(self, query: str) -> bool:
        """
        Validates the query for correct syntax and field codes.
        """
        # List of valid field codes
        valid_field_codes = ['TITLE', 'ABS', 'KEY', 'AUTH', 'AFFIL', 'SRCTITLE', 'DOCTYPE', 'LANGUAGE', 'PUBYEAR']

        # Extract all field codes used in the query
        field_codes = re.findall(r'(\b[A-Z]+)\(', query)
        for code in field_codes:
            if code not in valid_field_codes:
                self.logger.error(f"Invalid field code used: {code}")
                return False

        # Check for balanced parentheses
        if query.count('(') != query.count(')'):
            self.logger.error("Unbalanced parentheses in the query.")
            return False

        return True

    def extract_content(self, response) -> str:
        """
        Extracts the text content from the LLM response.

        Parameters:
        - response: The response object from the LLM.

        Returns:
        - content (str): The extracted text content.
        """
        try:
            if isinstance(response, str):
                return response.strip()
            elif hasattr(response, 'content'):
                return response.content.strip()
            elif isinstance(response, dict) and 'content' in response:
                return response['content'].strip()
            else:
                self.logger.warning("Unexpected response format. Attempting to convert to string.")
                return str(response).strip()
        except Exception as e:
            self.logger.error(f"Error extracting content from response: {e}")
            return ""

    async def fetch_batch(self, session, url, headers, params):
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                self.logger.error(f"Error: {response.status} - {await response.text()}")
                return None

    async def execute_query_async(self, query: str, max_results: int = 10000) -> None:
        """
        Asynchronously executes the Scopus query and retrieves the results.
        """
        headers = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "application/json"
        }
        params = {
            "query": query,
            "start": 0,
            "count": 25,  # Increased batch size
            "view": "COMPLETE"
        }

        self.logger.info(f"Executing async query: {query}")

        articles = []
        total_results = 0

        async with aiohttp.ClientSession() as session:
            tasks = []
            while len(articles) < max_results:
                task = asyncio.ensure_future(self.fetch_batch(session, self.base_url, headers, params))
                tasks.append(task)
                responses = await asyncio.gather(*tasks)
                tasks = []

                for data in responses:
                    if data:
                        entries = data.get('search-results', {}).get('entry', [])
                        for entry in entries:
                            articles.append({
                                'Title': entry.get('dc:title', ''),
                                'Authors': entry.get('dc:creator', ''),
                                'Year': entry.get('prism:coverDate', '')[:4],
                                'DOI': entry.get('prism:doi', ''),
                                'Abstract': entry.get('dc:description', ''),
                                'Source': entry.get('prism:publicationName', ''),
                                'EID': entry.get('eid', ''),
                                'URL': entry.get('link', [{}])[0].get('@href', '')
                            })

                        total_results = int(data.get('search-results', {}).get('opensearch:totalResults', '0'))
                        self.logger.info(f"Retrieved {len(entries)} entries. Total results: {total_results}")

                        # Check if we've retrieved enough results or if there are no more results
                        if len(articles) >= max_results or len(entries) == 0:
                            break

                        # Update the starting index for the next request
                        params['start'] += params['count']

                        # Adjust count if approaching max_results
                        if len(articles) + params['count'] > max_results:
                            params['count'] = max_results - len(articles)
                    else:
                        break

        self.results = pd.DataFrame(articles[:max_results])

    def execute_query(self, query: str, max_results: int = 10000) -> None:
        """
        Synchronously executes the Scopus query by running the asynchronous method.
        """
        try:
            asyncio.run(self.execute_query_async(query, max_results))
        except Exception as e:
            self.logger.error(f"Error executing async query: {e}")
            self.results = pd.DataFrame()

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Returns the query results as a pandas DataFrame.
        """
        return self.results
