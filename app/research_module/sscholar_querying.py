import requests
import pandas as pd
import re
import json
import logging
from pydantic import BaseModel

class LLMResponse(BaseModel):
    content: str

class SemanticScholarQueryGenerator:
    def __init__(self, api_key, ollama_llm=None):
        """
        Initializes the SemanticScholarQueryGenerator.

        Parameters:
        - api_key (str): Your Semantic Scholar API key.
        - ollama_llm: An instance of the LLM for generating and improving queries.
        """
        self.api_key = api_key
        self.ollama_llm = ollama_llm
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

    def generate_query(self, topic, known_paper_ids=None):
        """
        Generates a Semantic Scholar query for the given topic.

        Parameters:
        - topic (str): The research topic.
        - known_paper_ids (list): Optional list of known paper IDs to tailor the query.

        Returns:
        - query (str): The generated query.
        """
        if not self.ollama_llm:
            raise ValueError("An LLM instance is required to generate the query.")

        if known_paper_ids:
            articles = self.fetch_article_info(known_paper_ids)
            if not articles:
                self.logger.error("No articles fetched with the provided paper IDs.")
                analysis_text = self.analyze_topic(topic)
            else:
                analysis_text = self.analyze_articles(articles)
        else:
            analysis_text = self.analyze_topic(topic)

        terms = self.extract_terms(analysis_text)
        query = self.construct_semantic_scholar_query(terms)

        self.last_query = query  # Store the last generated query
        return query

    def analyze_topic(self, topic):
        """
        Analyzes the topic using the LLM to extract relevant terms.

        Parameters:
        - topic (str): The research topic.

        Returns:
        - analysis (str): The analysis text from the LLM.
        """
        prompt = f"""Analyze the following research topic and extract key concepts, methodologies, and terminology that would be relevant for a Semantic Scholar search query.

Topic:
{topic}

Provide a concise analysis."""
        response = self.ollama_llm.invoke(prompt)
        analysis = self.extract_content(response)
        return analysis

    def extract_terms(self, analysis_text):
        """
        Extracts key terms from the analysis text provided by the LLM.

        Parameters:
        - analysis_text (str): The analysis text containing terms.

        Returns:
        - terms (list): A list of extracted terms.
        """
        prompt = f"""From the following analysis, extract key terms relevant for a Semantic Scholar search.

    Analysis:
    {analysis_text}

    Provide the terms in JSON format as:
    {{
        "keywords": ["term1", "term2", ...]
    }}

    Do not include any additional text or explanations."""
        response = self.ollama_llm.invoke(prompt)
        content = self.extract_content(response)
        print("LLM Output:", content)  # Debugging statement
        # Attempt to fix common JSON issues
        content_fixed = content.strip()
        if content_fixed.startswith("```json"):
            content_fixed = content_fixed[7:]
        if content_fixed.startswith("```"):
            content_fixed = content_fixed[3:]
        if content_fixed.endswith("```"):
            content_fixed = content_fixed[:-3]
        try:
            terms_dict = json.loads(content_fixed)
            terms = terms_dict.get("keywords", [])
            print("Extracted Terms:", terms)  # Debugging statement
            return terms
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse terms from LLM response: {e}")
            return []

    def construct_semantic_scholar_query(self, terms):
        """
        Constructs a Semantic Scholar query from the extracted terms.

        Parameters:
        - terms (list): List of terms.

        Returns:
        - query (str): The constructed Semantic Scholar query.
        """
        if not terms:
            self.logger.error("No terms provided to construct the query.")
            return ""

        # Join the terms with spaces to create a simple keyword query
        query = ' '.join(terms)
        print("Constructed Query:", query)  # Debugging statement
        return query

    def fetch_article_info(self, paper_ids):
        """
        Fetches article information for given paper IDs from Semantic Scholar.

        Parameters:
        - paper_ids (list): List of Semantic Scholar paper IDs.

        Returns:
        - articles (list): List of dictionaries containing article info.
        """
        # Filter out invalid paper IDs
        paper_ids = [pid for pid in paper_ids if pid and isinstance(pid, str)]
        if not paper_ids:
            self.logger.error("No valid paper IDs provided.")
            return []

        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/batch"
            params = {
                "fields": "paperId,title,abstract,authors,year,venue"
            }
            data = {
                "ids": paper_ids
            }
            response = requests.post(url, headers=self.headers, params=params, json=data)

            if response.status_code == 200:
                articles = response.json()
                return articles
            else:
                self.logger.error(f"Error fetching article info: {response.status_code} {response.text}")
                return []
        except Exception as e:
            self.logger.error(f"Error fetching article info: {str(e)}")
            return []

    def analyze_articles(self, articles):
        """
        Analyzes articles to extract key terms using the LLM.

        Parameters:
        - articles (list): List of article dictionaries.

        Returns:
        - analysis (str): Analysis result from the LLM.
        """
        combined_text = "\n\n".join([
            f"Title: {article.get('title', '')}\nAbstract: {article.get('abstract', '')}"
            for article in articles if article.get('title') and article.get('abstract')
        ])

        if not combined_text:
            self.logger.error("No valid articles to analyze.")
            return ""

        prompt = f"""Analyze the following scientific articles and extract key concepts, methodologies, and terminology that would be relevant for a Semantic Scholar search query:

{combined_text}

Provide a concise summary of the main research topic and list key terms that should be included in a Semantic Scholar search query to find similar articles."""
        response = self.ollama_llm.invoke(prompt)
        analysis = self.extract_content(response)
        return analysis

    def execute_query(self, query, limit=100):
        """
        Executes the Semantic Scholar query and retrieves the results.

        Parameters:
        - query (str): The Semantic Scholar query to execute.
        - limit (int): Maximum number of results to retrieve.
        """
        try:
            params = {
                "query": query,
                "limit": limit,
                "fields": "paperId,title,abstract,authors,year,venue,embedding"
            }

            # Apply date filters
            filters = []
            if self.from_year and self.to_year:
                filters.append(f"year:{self.from_year}-{self.to_year}")
            elif self.from_year:
                filters.append(f"year:{self.from_year}-")
            elif self.to_year:
                filters.append(f"year:-{self.to_year}")

            if filters:
                params['filter'] = ','.join(filters)

            response = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                headers=self.headers,
                params=params
            )

            if response.status_code == 200:
                data = response.json()
                papers = data.get('data', [])
                self.results = self.process_papers(papers)
            else:
                self.logger.error(f"Error executing query: {response.status_code} {response.text}")
                self.results = pd.DataFrame()
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
        embeddings = []

        for paper in papers:
            titles.append(paper.get('title', ''))
            abstracts.append(paper.get('abstract', ''))
            authors = [author.get('name', '') for author in paper.get('authors', [])]
            authors_list.append(', '.join(authors))
            years.append(paper.get('year', ''))
            venues.append(paper.get('venue', ''))
            paper_ids.append(paper.get('paperId', ''))
            embedding = paper.get('embedding', {}).get('vector', None)
            embeddings.append(embedding)

        df = pd.DataFrame({
            "Paper ID": paper_ids,
            "Title": titles,
            "Abstract": abstracts,
            "Authors": authors_list,
            "Year": years,
            "Venue": venues,
            "Embedding": embeddings
        })
        return df

    def get_results_dataframe(self):
        """
        Returns the query results as a pandas DataFrame.

        Returns:
        - DataFrame containing the results.
        """
        return self.results

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
                return response
            elif hasattr(response, 'content'):
                return response.content
            elif isinstance(response, dict) and 'content' in response:
                return response['content']
            else:
                self.logger.warning("Unexpected response format. Attempting to convert to string.")
                return str(response)
        except Exception as e:
            self.logger.error(f"Error extracting content from response: {e}")
            return ""


from langchain_ollama.chat_models import ChatOllama
import os
from dotenv import load_dotenv
import requests
import json
# Load environment variables
dotenv_path = '/Users/fernando/Documents/Research/academatepy/.env'
load_dotenv(dotenv_path)

# Authenticate and get JWT token if required
auth_response = requests.post(
    os.getenv('AUTH_URL'),
    json={'email': os.getenv('ollama_user'), 'password': os.getenv('ollama_pw')}
)
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

# Get the API key from environment variable
api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

# Create an instance of the SemanticScholarQueryGenerator
semantic_scholar_query_gen = SemanticScholarQueryGenerator(api_key, ollama_llm)

# Optionally set date filters
semantic_scholar_query_gen.set_date_filter(from_year=2010)

# Generate a query for a topic
topic = "The role of artificial intelligence in drug discovery"
# No known paper IDs
query = semantic_scholar_query_gen.generate_query(topic)

print(f"Generated Query:\n{query}")

# Execute the query
semantic_scholar_query_gen.execute_query(query)

# Get the results as a DataFrame
results_df = semantic_scholar_query_gen.get_results_dataframe()

print(results_df.head())
