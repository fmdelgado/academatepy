import requests
import pandas as pd
import re
import json
import logging

class SemanticScholarQueryGenerator:
    def __init__(self, api_key, llm=None, two_step_search_queries=False):
        """
        Initializes the SemanticScholarQueryGenerator.

        Parameters:
        - api_key (str): Your Semantic Scholar API key.
        - llm: An instance of the LLM for generating and improving queries.
        """
        self.api_key = api_key
        self.llm = llm
        self.two_step_search_queries = two_step_search_queries
        self.headers = {"x-api-key": self.api_key}
        self.from_year = None
        self.to_year = None
        self.results = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def set_date_filter(self, from_year=None, to_year=None):
        """
        Sets the date filter for the Semantic Scholar query.

        Parameters:
        - from_year (int): Start year for the date filter.
        - to_year (int): End year for the date filter.
        """
        self.from_year = from_year
        self.to_year = to_year

    def generate_query(self, topic):
        """
        Generates a Semantic Scholar query for the given topic.
        """
        if not self.llm:
            raise ValueError("An LLM instance is required to generate the query.")

        for attempt in range(2):  # Limit the number of attempts to prevent infinite loops
            if self.two_step_search_query:
                query = self.generate_semantic_scholar_query_given_keywords(topic)
            else:
                query = self.generate_semantic_scholar_query(topic)
            query = self.clean_query(query)

            # Validate the query
            if self.validate_query(query):
                self.last_query = query  # Store the last generated query
                return query
            else:
                self.logger.warning("Generated query is invalid. Re-prompting the model.")
                # Modify the prompt to emphasize the error
                self.generate_semantic_scholar_query(topic, error=True)

        raise ValueError("Failed to generate a valid query after multiple attempts.")

    def generate_semantic_scholar_query(self, topic: str, error=False) -> str:
        """
        Generates a Semantic Scholar query using the LLM.
        """
        error_message = "The previous query was invalid due to syntax errors or missing markers. Please follow the instructions carefully."

        prompt = f"""You are an expert in creating Semantic Scholar search queries.

    Generate a comprehensive search query for the following research topic:

    Topic:
    {topic}

    {error_message if error else ''}
    
    You are an expert in creating Semantic Scholar search queries.

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

    **Example of an incorrect response (do not do this):**

    Here is your query:
    <START_QUERY>
    ("term1" | "term1 synonym") + ("term2" | "term2 synonym")
    <END_QUERY>

    Provide your query below:
    """
        response = self.llm.invoke(prompt)
        content = self.extract_content(response)
        query = self.extract_query_from_markers(content)
        query = self.clean_query(query)
        return query
    
    def generate_semantic_scholar_query_given_keywords(self, keywords: str, error=False) -> str:
        """
        Generates a Semantic Scholar query using the LLM.
        """
        error_message = "The previous query was invalid due to syntax errors or missing markers. Please follow the instructions carefully."

        prompt = f"""You are an expert in creating Semantic Scholar search queries.

    Generate a comprehensive search query for the following research topic:

    Topic:
    {keywords}

    {error_message if error else ''}
    
    You are an expert in creating Semantic Scholar search queries.

    Generate a comprehensive search query for the following research topic:

    Topic:
    {keywords}

    **Instructions:**
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

    **Example of an incorrect response (do not do this):**

    Here is your query:
    <START_QUERY>
    ("term1" | "term1 synonym") + ("term2" | "term2 synonym")
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
        If markers are not found, attempts to extract the query heuristically.
        """
        match = re.search(r'<START_QUERY>(.*?)<END_QUERY>', content, re.DOTALL | re.IGNORECASE)
        if match:
            query = match.group(1).strip()
        else:
            self.logger.warning("Markers not found in LLM output. Attempting to extract query heuristically.")
            # Heuristic extraction: Assume the query is the first code block or first non-empty line
            code_blocks = re.findall(r'```.*?```', content, re.DOTALL)
            if code_blocks:
                query = code_blocks[0].strip('`').strip()
            else:
                lines = content.strip().split('\n')
                query = lines[0].strip()
        return query

    def clean_query(self, query: str) -> str:
        """
        Cleans and formats the generated query.
        """
        if not query:
            return ""

        # Remove any accidental markers or extra text
        query = re.sub(r'<.*?>', '', query)

        # Replace fancy quotes with straight quotes
        query = query.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")

        # Replace curly braces with parentheses
        query = query.replace('{', '(').replace('}', ')')

        # Remove any unsupported characters
        query = re.sub(r'[^\w\s\+\|\-\(\)\*\"\'\:\.]', '', query)

        # Remove extra spaces
        query = re.sub(r'\s+', ' ', query).strip()

        # Ensure balanced parentheses
        if query.count('(') != query.count(')'):
            self.logger.warning("Unbalanced parentheses in the query.")

        return query

    def extract_content(self, response):
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
        Executes the Semantic Scholar query and retrieves the results.
        """
        try:
            params = {
                "query": query,
                "fields": "paperId,title,abstract,authors,year,venue,externalIds",  # Added 'externalIds'
                "limit": 1000  # Maximum allowed per request
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
            token = None

            while total_results < max_results:
                if token:
                    params['token'] = token

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

                    if 'next' in data and data['next']:
                        token = data['next']
                    else:
                        break  # No more data
                else:
                    self.logger.error(f"Error executing query: {response.status_code} {response.text}")
                    break

                # Stop if we've reached the max_results
                if total_results >= max_results:
                    break

            self.results = self.process_papers(papers[:max_results])
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            self.results = pd.DataFrame()

    def process_papers(self, papers):
        """
        Processes the list of papers and returns a DataFrame.

        Parameters:
        - papers (list): List of paper dictionaries.

        Returns:
        - DataFrame containing the paper information.
        """
        titles = []
        abstracts = []
        authors_list = []
        years = []
        venues = []
        paper_ids = []
        dois = []

        for paper in papers:
            titles.append(paper.get('title', ''))
            abstracts.append(paper.get('abstract', ''))
            authors = [author.get('name', '') for author in paper.get('authors', [])]
            authors_list.append(', '.join(authors))
            years.append(paper.get('year', ''))
            venues.append(paper.get('venue', ''))
            paper_ids.append(paper.get('paperId', ''))
            external_ids = paper.get('externalIds', {})
            doi = external_ids.get('DOI', '')
            dois.append(doi)

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

    def validate_query(self, query: str) -> bool:
        """
        Validates the query for correct syntax.
        """
        # Check for unbalanced parentheses
        if query.count('(') != query.count(')'):
            self.logger.error("Unbalanced parentheses in the query.")
            return False

        # Check for unsupported characters (e.g., curly braces)
        if '{' in query or '}' in query:
            self.logger.error("Unsupported characters in the query.")
            return False

        # Check for invalid operators or characters
        invalid_chars = re.findall(r'[^\w\s\+\|\-\(\)\*\"\'\:\.]', query)
        if invalid_chars:
            self.logger.error(f"Unsupported characters in the query: {set(invalid_chars)}")
            return False

        return True

    def get_results_dataframe(self):
        """
        Returns the query results as a pandas DataFrame.

        Returns:
        - DataFrame containing the results.
        """
        return self.results
