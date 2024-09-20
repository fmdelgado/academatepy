import os
import pandas as pd
import re
import logging
from typing import List, Optional
from pydantic import BaseModel
from pybliometrics.scopus import ScopusSearch, AbstractRetrieval
from pybliometrics.scopus.exception import ScopusQueryError
from tqdm import tqdm
import json

class LLMResponse(BaseModel):
    content: str

class ScopusQueryGenerator:
    def __init__(self, api_key: str, llm=None):
        self.api_key = api_key
        self.llm = llm
        self.from_year = None
        self.to_year = None
        self.results = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def set_date_filter(self, from_year: Optional[int] = None, to_year: Optional[int] = None):
        self.from_year = from_year
        self.to_year = to_year

    def generate_query(self, topic: str, known_ids: Optional[List[str]] = None) -> str:
        if not self.llm:
            raise ValueError("An LLM instance is required to generate the query.")

        if known_ids:
            articles = self.fetch_article_info(known_ids)
            analysis = self.analyze_articles(articles)
        else:
            # If no known IDs, use the topic directly
            analysis = self.analyze_topic(topic)

        query = self.construct_query_from_analysis(analysis)

        # Apply date filters if set
        if self.from_year or self.to_year:
            date_filter = self.get_date_filter()
            query = f"({query}) AND {date_filter}"

        self.last_query = query  # Store the last generated query
        return query

    def analyze_topic(self, topic: str) -> dict:
        """
        Analyzes the topic to extract key terms using the LLM.

        Returns:
        - analysis (dict): A dictionary with lists of key terms for different concepts.
        """
        prompt = f"""Analyze the following research topic and extract key concepts, methodologies, and terminology that would be relevant for a Scopus advanced search query.

    Topic:
    {topic}

    Provide the key terms and synonyms in a JSON object with the following structure:

    {{
      "Concept1": ["term1", "term1 synonym", "term1 related term"],
      "Concept2": ["term2", "term2 synonym", "term2 related term"],
      ...
    }}

    Do not include any additional text or explanations.
    """

        response = self.llm.invoke(prompt)
        analysis = self.extract_json(response)
        return analysis

    def construct_query_from_analysis(self, analysis: dict) -> str:
        """
        Constructs a Scopus advanced search query from the analysis data.

        Parameters:
        - analysis (dict): Dictionary containing key terms.

        Returns:
        - query (str): The constructed Scopus query.
        """

        if not analysis:
            self.logger.error("Analysis data is empty. Cannot construct query.")
            return ""

        concept_queries = []
        for concept_terms in analysis.values():
            # Ensure terms are in quotes if they contain spaces
            terms = [f'"{term}"' if ' ' in term else term for term in concept_terms]
            concept_query = " OR ".join(terms)
            concept_queries.append(f'TITLE-ABS-KEY({concept_query})')

        query = " AND ".join(concept_queries)
        return query

    def get_date_filter(self) -> str:
        date_filters = []
        if self.from_year:
            date_filters.append(f"PUBYEAR AFT {self.from_year - 1}")
        if self.to_year:
            date_filters.append(f"PUBYEAR BEF {self.to_year + 1}")
        return ' AND '.join(date_filters)

    def execute_query(self, query: str, max_results: int = 10000) -> None:
        try:
            self.logger.info(f"Executing query: {query}")
            s = ScopusSearch(query, subscriber=True, verbose=False, view='STANDARD')
            if s.results is None:
                self.logger.error("No results found.")
                self.results = pd.DataFrame()
                return
            total_results = len(s.results)
            self.logger.info(f"Found {total_results} results")

            if total_results == 0:
                self.results = pd.DataFrame()
                return

            articles = []
            for result in tqdm(s.results[:max_results], desc="Fetching results"):
                try:
                    abstract = AbstractRetrieval(result.eid, view='FULL')
                    articles.append({
                        'Title': result.title,
                        'Authors': ', '.join(result.author_names) if result.author_names else '',
                        'Year': result.coverDate[:4] if result.coverDate else '',
                        'DOI': result.doi,
                        'Citations': result.citedby_count,
                        'Abstract': abstract.abstract,
                        'Keywords': '; '.join(abstract.authkeywords or []),
                        'Journal': result.publicationName,
                        'Volume': result.volume,
                        'Issue': result.issueIdentifier,
                        'Pages': result.pageRange,
                        'ISSN': result.issn,
                        'Document Type': result.subtypeDescription,
                        'Source ID': result.source_id,
                        'EID': result.eid
                    })
                except Exception as e:
                    self.logger.error(f"Error retrieving details for {result.eid}: {str(e)}")
            self.results = pd.DataFrame(articles)
        except ScopusQueryError as e:
            self.logger.error(f"Error executing query: {e}")
            self.results = pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.results = pd.DataFrame()

    def get_results_dataframe(self) -> pd.DataFrame:
        return self.results

    def fetch_article_info(self, scopus_ids: List[str]) -> List[dict]:
        articles = []
        for scopus_id in scopus_ids:
            try:
                abstract = AbstractRetrieval(scopus_id, view='FULL')
                articles.append({
                    'scopus_id': scopus_id,
                    'Title': abstract.title,
                    'Abstract': abstract.abstract
                })
            except Exception as e:
                self.logger.error(f"Error fetching article {scopus_id}: {str(e)}")
        return articles

    def analyze_articles(self, articles: List[dict]) -> dict:
        """
        Analyzes articles to extract key terms using the LLM.

        Returns:
        - analysis (dict): A dictionary with lists of key terms for different concepts.
        """
        combined_text = "\n\n".join([f"Title: {a['Title']}\nAbstract: {a['Abstract']}" for a in articles])

        prompt = f"""Analyze the following scientific articles and extract key concepts, methodologies, and terminology that would be relevant for a Scopus advanced search query.

    {combined_text}

    Provide the key terms and synonyms in a JSON object with the following structure:

    {{
      "Concept1": ["term1", "term1 synonym", "term1 related term"],
      "Concept2": ["term2", "term2 synonym", "term2 related term"],
      ...
    }}

    Do not include any additional text or explanations.
    """

        response = self.llm.invoke(prompt)
        content = self.extract_content(response)
        print("LLM raw output:", content)  # Add this line
        analysis = self.extract_json(response)
        return analysis

    def extract_json(self, response) -> dict:
        """
        Extracts a JSON object from the LLM response.

        Returns:
        - content (dict): The extracted JSON data.
        """
        try:
            content = self.extract_content(response)
            json_data = re.search(r'{.*}', content, re.DOTALL)
            if json_data:
                return json.loads(json_data.group())
            else:
                self.logger.error("No JSON object found in LLM response.")
                return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decoding error: {e}")
            return {}

    def generate_context_aware_query(self, topic: str, article_analysis: str) -> str:
        prompt = f"""Generate a Scopus advanced search query for the following topic:

{topic}

Consider the following analysis of relevant articles:

{article_analysis}

Guidelines:
1. Use field codes like TITLE(), ABS(), KEY(), TITLE-ABS-KEY() without quotes or colons.
2. Use boolean operators (AND, OR, NOT) to create effective query structures.
3. Include relevant synonyms and related terms from the article analysis.
4. Balance specificity and sensitivity.
5. Do not include any additional text, explanations, or notes.
6. Provide ONLY the query, formatted correctly for Scopus advanced search.
7. Do not include any special characters like colons (:), quotation marks (") around field codes.
8. Do not include JSON or key-value pairs.

Your response should be the Scopus query, in this format:

(TITLE-ABS-KEY(term1 OR "term1 synonym") AND TITLE-ABS-KEY(term2 OR "term2 synonym"))

Ensure that the query is syntactically correct for Scopus advanced search.
"""

        response = self.llm.invoke(prompt)
        query = self.extract_content(response)
        query = self.clean_query(query)
        return query

    def generate_scopus_query(self, topic: str) -> str:
        prompt = f"""Generate a comprehensive Scopus advanced search query for this topic: {topic}

Guidelines:
1. Use field codes: TITLE(), ABS(), KEY(), TITLE-ABS-KEY() without quotes or colons.
2. Use boolean operators: AND, OR, NOT.
3. Include synonyms and related terms.
4. Balance specificity and sensitivity.
5. Do not include any additional text, explanations, or notes.
6. Provide ONLY the query, formatted correctly for Scopus advanced search.
7. Do not include any special characters like colons (:), quotation marks (") around field codes.
8. Do not include JSON or key-value pairs.

Your response should be the Scopus query, in this format:

(TITLE-ABS-KEY(term1 OR "term1 synonym") AND TITLE-ABS-KEY(term2 OR "term2 synonym"))

Ensure that the query is syntactically correct for Scopus advanced search.
"""

        response = self.llm.invoke(prompt)
        query = self.extract_content(response)
        query = self.clean_query(query)
        return query

    def clean_query(self, query: str) -> str:
        if not query:
            return ""

        # Remove leading and trailing whitespaces
        query = query.strip()

        # Remove any JSON-like structures or key-value pairs
        if ':' in query and not query.strip().startswith('TITLE'):
            query = query.split(':', 1)[-1].strip()

        # Remove any leading/trailing quotes or brackets
        query = re.sub(r'^[{\[\("]+', '', query)
        query = re.sub(r'[}\]\)"]+$', '', query)

        # Remove any extra annotations or notes if present
        query = re.sub(r'Notes?:.*', '', query, flags=re.IGNORECASE)

        # Ensure the query is properly formatted for Scopus advanced search
        query = query.strip()
        return query

    def extract_content(self, response) -> str:
        """
        Extracts the text content from the LLM response.

        Returns:
        - content (str): The extracted text content.
        """
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, dict) and 'content' in response:
            return response['content']
        elif isinstance(response, str):
            return response
        else:
            self.logger.warning("Unexpected response format. Attempting to convert to string.")
            return str(response)
#
#
# from langchain_ollama.chat_models import ChatOllama
# import os
# from dotenv import load_dotenv
# import requests
# import json
# import pybliometrics.scopus
# pybliometrics.scopus.init()
#
# # Authenticate and get JWT token if required
# dotenv_path = '/Users/fernando/Documents/Research/academatepy/.env'
# load_dotenv(dotenv_path)
#
# auth_response = requests.post(os.getenv('AUTH_URL'), json= {'email': os.getenv('ollama_user'), 'password': os.getenv('ollama_pw')})
# jwt = json.loads(auth_response.text)["token"]
# headers = {"Authorization": "Bearer " + jwt}
#
#
# # Create an instance of the LLM
# ollama_llm = ChatOllama(
#     base_url=os.getenv('OLLAMA_API_URL'),
#     model='mistral:v0.2',                # Replace with your LLM model
#     temperature=0.0,
#     client_kwargs={'headers': headers},
#     format="json"
# )
#
#
# # Your Elsevier Scopus API key
# scopus_api_key = os.getenv('SCOPUS_API_KEY')
#
# # Create an instance of the ScopusQueryGenerator
# scopus_query_gen = ScopusQueryGenerator(scopus_api_key, ollama_llm)
#
# # Optionally set date filters
# scopus_query_gen.set_date_filter(from_year=2010, to_year=2023)
#
# # Generate a query for a topic
# topic = "The role of artificial intelligence in drug discovery"
# known_ids = ["85089390111", "85082388177"]  # Example Scopus IDs (ensure these are valid)
#
# query = scopus_query_gen.generate_query(topic, known_ids=known_ids)
#
# query = scopus_query_gen.generate_query(topic)
#
# print(f"Generated Query:\n{query}")
#
# # Execute the query
# scopus_query_gen.execute_query(query)
#
# # Get the results as a DataFrame
# results_df = scopus_query_gen.get_results_dataframe()
#
# print(results_df.head())