import os
import pandas as pd
import re
import logging
from typing import List, Optional
import requests
import json
from pydantic import BaseModel

class EmbaseQueryGenerator:
    def __init__(self, api_key: str, llm=None):
        """
        Initializes the EmbaseQueryGenerator.

        Parameters:
        - api_key (str): Your Elsevier Embase API key.
        - llm: An instance of the LLM for generating and improving queries.
        """
        self.api_key = api_key
        self.llm = llm
        self.from_year = None
        self.to_year = None
        self.results = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.base_url = "https://api.elsevier.com/content/search/index:EMBASE"

    def set_date_filter(self, from_year: Optional[int] = None, to_year: Optional[int] = None):
        """
        Sets the date filter for the Embase query.

        Parameters:
        - from_year (int): Start year for the date filter.
        - to_year (int): End year for the date filter.
        """
        self.from_year = from_year
        self.to_year = to_year

    def generate_query(self, topic: str) -> str:
        """
        Generates an Embase advanced search query for the given topic.

        Parameters:
        - topic (str): The research topic.

        Returns:
        - query (str): The generated Embase query.
        """
        if not self.llm:
            raise ValueError("An LLM instance is required to generate the query.")

        query = self.generate_embase_query(topic)

        # Apply date filters if set
        if self.from_year or self.to_year:
            date_filter = self.get_date_filter()
            query = f"({query}) AND {date_filter}"

        return query

    def get_date_filter(self) -> str:
        date_filters = []
        if self.from_year:
            date_filters.append(f"PUBYEAR AFT {self.from_year - 1}")
        if self.to_year:
            date_filters.append(f"PUBYEAR BEF {self.to_year + 1}")
        return ' AND '.join(date_filters)

    def execute_query(self, query: str, max_results: int = 100) -> None:
        """
        Executes the Embase query and retrieves the results.

        Parameters:
        - query (str): The Embase query to execute.
        - max_results (int): Maximum number of results to retrieve.
        """
        headers = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "application/json"
        }
        params = {
            "query": query,
            "count": max_results,
            "sort": "relevance",
        }

        self.logger.info(f"Executing query: {query}")

        try:
            response = requests.get(self.base_url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                entries = data.get('search-results', {}).get('entry', [])
                articles = []
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
                self.results = pd.DataFrame(articles)
                total_results = data.get('search-results', {}).get('opensearch:totalResults', '0')
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

        Returns:
        - DataFrame containing the results.
        """
        return self.results

    def generate_embase_query(self, topic: str) -> str:
        """
        Generates an Embase advanced search query using the LLM.
        """
        prompt = f"""Generate a comprehensive Embase advanced search query for the following topic:
        {topic}
        Guidelines:
        1. Use field codes like TITLE(), ABS(), KEY(), TITLE-ABS-KEY() without quotes or colons.
        2. Use boolean operators (AND, OR, NOT) to create effective query structures.
        3. Include relevant synonyms and related terms.
        4. Balance specificity and sensitivity.
        5. Do not include any additional text, explanations, or notes.
        6. Provide ONLY the query, formatted correctly for Embase advanced search.
        7. Do not include any special characters like colons (:), quotation marks (") around field codes.
        8. Do not include JSON or key-value pairs.
    
        Your response should be the Embase query, in this format:
    
        (TITLE-ABS-KEY(term1 OR "term1 synonym") AND TITLE-ABS-KEY(term2 OR "term2 synonym"))
    
        Ensure that the query is syntactically correct for Embase advanced search.
        """

        response = self.llm.invoke(prompt)
        query = self.extract_content(response)
        query = self.clean_query(query)
        return query

    def clean_query(self, query: str) -> str:
        """
        Cleans and formats the generated query.
        """
        if not query:
            return ""

        # Remove any JSON-like structures or key-value pairs
        if ':' in query and not query.strip().startswith('TITLE'):
            # Attempt to extract the actual query
            match = re.search(r'"?query"?\s*:\s*"(.*?)"', query)
            if match:
                query = match.group(1)
            else:
                # Remove any key-value pairs
                query = re.sub(r'".+?":\s*".+?"', '', query)

        # Remove any leading/trailing quotes or braces
        query = query.strip()
        query = re.sub(r'^[{\[\("]+', '', query)
        query = re.sub(r'[}\]\)"]+$', '', query)

        # Remove any extra annotations or notes if present
        query = re.sub(r'Notes?:.*', '', query, flags=re.IGNORECASE)

        # Remove extraneous characters
        query = re.sub(r'["\']', '', query)
        query = query.strip()
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
#
#
from langchain_ollama.chat_models import ChatOllama
import os
from dotenv import load_dotenv
import requests
import json

# Authenticate and get JWT token if required
dotenv_path = '/Users/fernando/Documents/Research/academatepy/.env'
load_dotenv(dotenv_path)

auth_response = requests.post(os.getenv('AUTH_URL'), json= {'email': os.getenv('ollama_user'), 'password': os.getenv('ollama_pw')})
jwt = json.loads(auth_response.text)["token"]
headers = {"Authorization": "Bearer " + jwt}

# Create an instance of the LLM
ollama_llm = ChatOllama(
    base_url=os.getenv('OLLAMA_API_URL'),
    model='mistral:v0.2',
    temperature=0.0,
    client_kwargs={'headers': headers},
    format="json"
)

# Create an instance of the PubMedQueryGenerator
email = "your.email@example.com"
embase_api_key = os.getenv("EMBASE_API_KEY")  # Replace with your actual Embase API key

# Generate a query for a topic
topic = "The role of artificial intelligence in drug discovery"
known_pmids = ["33844136", "33099022"]  # Optional known PMIDs to tailor the query

models = ['reflection:70b', 'llama3:8b-instruct-fp16', 'llama3:latest', 'mistral:v0.2', 'mistral-nemo:latest', 'medllama2:latest', 'meditron:70b', 'meditron:7b', 'llama3.1:8b', 'llama3.1:70b', 'gemma:latest', 'phi3:latest', 'phi3:14b', 'gemma2:9b']
# models = ['mistral:v0.2']
for model_name in models:
    print(f"Testing model: {model_name}")
    # Create an instance of the LLM with the current model
    ollama_llm = ChatOllama(
        base_url=os.getenv('OLLAMA_API_URL'),
        model=model_name,
        temperature=0.0,
        client_kwargs={'headers': headers},
        format="json"
    )
    # Create a new instance of the EmbaseQueryGenerator
    embase_query_gen = EmbaseQueryGenerator(embase_api_key, ollama_llm)
    embase_query_gen.set_date_filter(from_year=2010)
    # Generate the query
    query = embase_query_gen.generate_embase_query(topic)
    print(f"Generated Query with {model_name}:\n{query}\n")
    # Optionally, you can test executing the query and handle exceptions
    try:
        embase_query_gen.execute_query(query)
        results_df = embase_query_gen.get_results_dataframe()
        print(f"Results with {model_name}:\n{results_df.head()}\n")
    except Exception as e:
        print(f"Error with model {model_name}: {e}\n")
