import pandas as pd
import re
import logging
from typing import Optional
import requests

class EmbaseQueryGenerator:
    def __init__(self, api_key: str, llm=None, two_step_search_query=False):
        """
        Initializes the EmbaseQueryGenerator.

        Parameters:
        - api_key (str): Your Elsevier Embase API key.
        - llm: An instance of the LLM for generating and improving queries.
        - two_step_search_query (bool): Whether to use a two-step search query.
        """
        self.api_key = api_key
        self.llm = llm
        self.two_step_search_query = two_step_search_query
        self.from_year = None
        self.to_year = None
        self.results = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.base_url = "https://api.elsevier.com/content/embase/article"

    def set_date_filter(self, from_year: Optional[int] = None, to_year: Optional[int] = None):
        """
        Sets the date filter for the Embase query.
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

    def generate_query(self, topic: str) -> str:
        """
        Generates an Embase advanced search query for the given topic.
        """
        if not self.llm:
            raise ValueError("An LLM instance is required to generate the query.")

        if self.two_step_search_query:
            query = self.generate_embase_query_given_keywords(topic)
        else:
            query = self.generate_embase_query(topic)

        # Apply date filters if set
        if self.from_year or self.to_year:
            date_filter = self.get_date_filter()
            query = f"({query}) AND {date_filter}"

        return query

    def generate_embase_query(self, topic: str) -> str:
        """
        Generates an Embase Command Language query using the LLM.
        """
        prompt = f"""You are an expert in creating Embase Command Language search queries.

    Generate a comprehensive Embase search query in Command Language syntax for the following topic:

    Topic:
    {topic}

    Requirements:
    1. Identify the main concepts in the topic.
    2. For each concept, include synonyms and related terms.
    3. Use Emtree terms where appropriate, and include the appropriate field codes (e.g., '/de' for descriptors, '/exp' for exploded terms).
    4. Use Boolean operators (AND, OR, NOT) to create effective query structures.
    5. Balance specificity and sensitivity.
    6. Do not include any additional text, explanations, or notes.
    7. Provide ONLY the query, formatted correctly for Embase Command Language.
    8. Enclose your query between the markers <START_QUERY> and <END_QUERY>. Failure to do so will result in an invalid query.

    Example:

    <START_QUERY>
    ('term1'/exp OR 'term1 synonym'/exp) AND ('term2'/exp OR 'term2 synonym'/exp)
    <END_QUERY>

    Provide your query below:
    """

        response = self.llm.invoke(prompt)
        content = self.extract_content(response)
        query = self.extract_query_from_markers(content)
        query = self.clean_query(query)
        return query

    def generate_embase_query_given_keywords(self, keywords: str) -> str:
        """
        Generates an Embase Command Language query using the LLM.
        """
        prompt = f"""You are an expert in creating Embase Command Language search queries.

    Generate a comprehensive Embase search query in Command Language syntax for the following topic:

    Topic:
    {keywords}

    Requirements:
    1. Use Emtree terms where appropriate, and include the appropriate field codes (e.g., '/de' for descriptors, '/exp' for exploded terms).
    2. Use Boolean operators (AND, OR, NOT) to create effective query structures.
    3. Balance specificity and sensitivity.
    4. Do not include any additional text, explanations, or notes.
    5. Provide ONLY the query, formatted correctly for Embase Command Language.
    6. Enclose your query between the markers <START_QUERY> and <END_QUERY>. Failure to do so will result in an invalid query.

    Example:

    <START_QUERY>
    ('term1'/exp OR 'term1 synonym'/exp) AND ('term2'/exp OR 'term2 synonym'/exp)
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

        # Remove any accidental markers or extra text
        query = query.strip()

        # Replace fancy quotes with straight quotes
        query = query.replace('“', "'").replace('”', "'").replace('‘', "'").replace('’', "'")

        # Replace multiple spaces with a single space
        query = re.sub(r'\s+', ' ', query).strip()

        return query

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

    def execute_query(self, query: str, max_results: int = 10000) -> None:
        """
        Executes the Embase query and retrieves the results.
        """
        headers = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "application/json"
        }
        params = {
            "query": query,
            "start": "1",
            "count": str(max_results),
            "sort": "relevance",
        }

        self.logger.info(f"Executing query: {query}")

        try:
            response = requests.get(self.base_url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                entries = data.get('abstracts-retrieval-response', {}).get('items', [])
                articles = []
                for entry in entries:
                    coredata = entry.get('coredata', {})
                    articles.append({
                        'Title': coredata.get('dc:title', ''),
                        'Authors': coredata.get('dc:creator', ''),
                        'Year': coredata.get('prism:coverDate', '')[:4],
                        'DOI': coredata.get('prism:doi', ''),
                        'Abstract': coredata.get('dc:description', ''),
                        'Source': coredata.get('prism:publicationName', ''),
                        'EID': coredata.get('eid', ''),
                        'URL': coredata.get('link', [{}])[0].get('@href', '')
                    })
                self.results = pd.DataFrame(articles)
                total_results = data.get('abstracts-retrieval-response', {}).get('opensearch:totalResults', '0')
                self.logger.info(f"Found {total_results} results")
            else:
                self.logger.error(f"Error: {response.status_code} - {response.text}")
                self.results = pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.results = pd.DataFrame()

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Returns the query results as a pandas DataFrame.
        """
        return self.results
