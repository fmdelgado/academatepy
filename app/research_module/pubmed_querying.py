
import pandas as pd
from Bio import Entrez
import re
from pydantic import BaseModel
import logging
import json
class LLMResponse(BaseModel):
    content: str

class PubMedQueryGenerator:
    def __init__(self, email, api_key, ollama_llm=None):
        """
        Initializes the PubMedQueryGenerator.

        Parameters:
        - email (str): Your email address (required by NCBI).
        - api_key (str): Your NCBI API key.
        - ollama_llm: An instance of the LLM for generating and improving queries.
        """
        self.email = email
        self.api_key = api_key
        self.ollama_llm = ollama_llm
        Entrez.email = self.email
        Entrez.api_key = self.api_key
        self.from_year = None
        self.to_year = None
        self.results = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def set_date_filter(self, from_year=None, to_year=None):
        """
        Sets the date filter for the PubMed query.

        Parameters:
        - from_year (int): Start year for the date filter.
        - to_year (int): End year for the date filter.
        """
        self.from_year = from_year
        self.to_year = to_year

    def generate_query(self, topic, known_pmids=None):
        """
        Generates a PubMed query for the given topic.

        Parameters:
        - topic (str): The research topic.
        - known_pmids (list): Optional list of known PubMed IDs to tailor the query.

        Returns:
        - query (str): The generated PubMed query.
        """
        if not self.ollama_llm:
            raise ValueError("An LLM instance is required to generate the query.")

        if known_pmids:
            articles = self.fetch_article_info(known_pmids)
            analysis_text = self.analyze_articles(articles)
        else:
            analysis_text = self.analyze_topic(topic)

        terms = self.extract_terms(analysis_text)
        query = self.construct_pubmed_query(terms)

        # Apply date filters if set
        if self.from_year or self.to_year:
            date_filter = self.get_date_filter()
            query = f"({query}) AND {date_filter}"

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
        prompt = f"""Analyze the following research topic and extract key concepts, methodologies, and terminology that would be relevant for a PubMed search query.

    Topic:
    {topic}

    Provide a concise analysis."""
        response = self.ollama_llm.invoke(prompt)
        analysis = self.extract_content(response)
        return analysis

    def get_date_filter(self):
        from_year = self.from_year if self.from_year else "1000"
        to_year = self.to_year if self.to_year else "3000"
        date_filter = f'("{from_year}/01/01"[Date - Publication] : "{to_year}/12/31"[Date - Publication])'
        return date_filter

    def execute_query(self, query, max_results=10000):
        """
        Executes the PubMed query and retrieves the results.

        Parameters:
        - query (str): The PubMed query to execute.
        - max_results (int): Maximum number of results to retrieve.
        """
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(handle)
            pmids = record["IdList"]
            if not pmids:
                self.results = pd.DataFrame()
                return
            articles = self.fetch_article_info(pmids)
            self.results = pd.DataFrame(articles)
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            self.results = pd.DataFrame()

    def get_results_dataframe(self):
        """
        Returns the query results as a pandas DataFrame.

        Returns:
        - DataFrame containing the results.
        """
        return self.results

    def fetch_article_info(self, pmids):
        """
        Fetches article information for given PMIDs.

        Parameters:
        - pmids (list): List of PubMed IDs.

        Returns:
        - articles (list): List of dictionaries containing article info.
        """
        try:
            batch_size = 200  # NCBI recommends fetching in batches of 200 or less
            articles = []
            for start in range(0, len(pmids), batch_size):
                end = min(start + batch_size, len(pmids))
                batch_pmids = pmids[start:end]
                handle = Entrez.efetch(db="pubmed", id=",".join(batch_pmids), rettype="medline", retmode="xml")
                records = Entrez.read(handle)
                for record in records['PubmedArticle']:
                    medline_citation = record.get('MedlineCitation', {})
                    article = medline_citation.get('Article', {})
                    journal = article.get('Journal', {})
                    journal_issue = journal.get('JournalIssue', {})
                    pagination = article.get('Pagination', {})
                    pub_date = journal_issue.get('PubDate', {})
                    authors_list = article.get('AuthorList', [])
                    article_ids = record.get('PubmedData', {}).get('ArticleIdList', [])
                    mesh_headings = medline_citation.get('MeshHeadingList', [])

                    # PMID
                    pmid = medline_citation.get('PMID', '')

                    # Title
                    title = article.get('ArticleTitle', '')

                    # Abstract
                    abstract_list = article.get('Abstract', {}).get('AbstractText', [])
                    abstract = ' '.join(abstract_list)

                    # Authors
                    authors = []
                    for author in authors_list:
                        last_name = author.get('LastName', '')
                        fore_name = author.get('ForeName', '')
                        full_name = ' '.join([fore_name, last_name]).strip()
                        if full_name:
                            authors.append(full_name)
                    authors_str = ', '.join(authors)

                    # Year
                    year = pub_date.get('Year', '')
                    if not year and 'MedlineDate' in pub_date:
                        year = pub_date['MedlineDate'].split(' ')[0]

                    # Journal
                    journal_title = journal.get('Title', '')

                    # Volume
                    volume = journal_issue.get('Volume', '')

                    # Issue
                    issue = journal_issue.get('Issue', '')

                    # Pages
                    pages = pagination.get('MedlinePgn', '')

                    # DOI
                    doi = ''
                    # First, check in the ArticleIdList in PubmedData
                    article_id_list = record.get('PubmedData', {}).get('ArticleIdList', [])
                    for article_id in article_id_list:
                        if article_id.attributes.get('IdType') == 'doi':
                            doi = str(article_id)
                            break
                    # If not found, check in ArticleIdList in MedlineCitation
                    if not doi:
                        article_id_list = medline_citation.get('ArticleIdList', [])
                        for article_id in article_id_list:
                            if article_id.attributes.get('IdType') == 'doi':
                                doi = str(article_id)
                                break
                    # If still not found, check in ELocationID
                    if not doi:
                        elocation_ids = article.get('ELocationID', [])
                        if not isinstance(elocation_ids, list):
                            elocation_ids = [elocation_ids]
                        for eloc in elocation_ids:
                            if eloc.attributes.get('EIdType') == 'doi':
                                doi = str(eloc)
                                break

                    # ISSN
                    issn = journal.get('ISSN', '')

                    # Keywords (if available)
                    keywords_list = medline_citation.get('KeywordList', [])
                    keywords = []
                    for keyword_group in keywords_list:
                        for keyword in keyword_group:
                            keywords.append(str(keyword))
                    keywords_str = ', '.join(keywords)

                    # Document Type (Publication Type)
                    publication_types = article.get('PublicationTypeList', [])
                    doc_type = ', '.join([pt for pt in publication_types])

                    articles.append({
                        "PMID": pmid,
                        "Title": title,
                        "Abstract": abstract,
                        "Authors": authors_str,
                        "Year": year,
                        "Journal": journal_title,
                        "Volume": volume,
                        "Issue": issue,
                        "Pages": pages,
                        "DOI": doi,
                        "ISSN": issn,
                        "Keywords": keywords_str,
                        "Document Type": doc_type
                    })
            return articles
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
        combined_text = "\n\n".join([f"Title: {a['Title']}\nAbstract: {a['Abstract']}" for a in articles])

        prompt = f"""Analyze the following scientific articles and extract key concepts, methodologies, and terminology that would be relevant for a PubMed search query:

{combined_text}

Provide a concise summary of the main research topic and list key terms that should be included in a PubMed search query to find similar articles."""

        response = self.ollama_llm.invoke(prompt)

        # Extract the text content of the response
        analysis = self.extract_content(response)
        return analysis

    def generate_context_aware_query(self, topic, article_analysis):
        """
        Generates a context-aware PubMed query using the LLM.

        Parameters:
        - topic (str): The research topic.
        - article_analysis (str): Analysis of relevant articles.

        Returns:
        - query (str): The generated PubMed query.
        """
        prompt = f"""Generate a comprehensive PubMed search query for the following topic:

{topic}

Consider the following analysis of relevant articles:

{article_analysis}

Guidelines:
1. Use a combination of MeSH terms and free text searches.
2. Use field tags like [Title/Abstract], [MeSH Terms], etc.
3. Use boolean operators (AND, OR, NOT) to create effective query structures.
4. Include relevant synonyms and related terms from the article analysis.
5. Balance specificity and sensitivity.
6. Include a date range for publications since 2010.

Use this exact format for the date range:
("2010/01/01"[Date - Publication] : "3000"[Date - Publication])

Your query should look like this:
((term1[Title/Abstract] OR "term1"[MeSH Terms]) AND (term2[Title/Abstract] OR "term2"[MeSH Terms]) AND ("2010/01/01"[Date - Publication] : "3000"[Date - Publication]))

Provide ONLY the query, no explanations. Start with (( and end with ))."""

        response = self.ollama_llm.invoke(prompt)

        # Extract the text content of the response
        query = self.extract_content(response)

        query = self.clean_query(query)
        return query

    def generate_pubmed_query(self, topic):
        # Analyze the topic to extract terms
        analysis = self.analyze_topic(topic)
        terms = self.extract_terms(analysis)

        # Construct the query
        query = self.construct_pubmed_query(terms)
        return query

    def extract_terms(self, analysis_text):
        """
        Extracts key terms and MeSH terms from the analysis text provided by the LLM.

        Parameters:
        - analysis_text (str): The analysis text containing terms.

        Returns:
        - terms (dict): A dictionary with 'keywords' and 'MeSH' as keys.
        """
        prompt = f"""From the following analysis, extract key terms and MeSH terms relevant for PubMed search.

    Analysis:
    {analysis_text}

    Provide the terms in JSON format as:
    {{
        "keywords": ["term1", "term2", ...],
        "MeSH": ["MeSH term1", "MeSH term2", ...]
    }}

    Do not include any additional text or explanations."""

        response = self.ollama_llm.invoke(prompt)
        content = self.extract_content(response)
        try:
            terms = json.loads(content)
            return terms
        except json.JSONDecodeError:
            self.logger.error("Failed to parse terms from LLM response.")
            return {"keywords": [], "MeSH": []}

    def construct_pubmed_query(self, terms):
        """
        Constructs a PubMed query from the extracted terms.

        Parameters:
        - terms (dict): Dictionary containing 'keywords' and 'MeSH' terms.

        Returns:
        - query (str): The constructed PubMed query.
        """
        keyword_queries = [f'"{term}"[Title/Abstract]' for term in terms.get('keywords', [])]
        mesh_queries = [f'"{term}"[MeSH Terms]' for term in terms.get('MeSH', [])]

        combined_queries = keyword_queries + mesh_queries
        if not combined_queries:
            self.logger.error("No terms provided to construct the query.")
            return ""

        query = " OR ".join(combined_queries)
        query = f"({query})"
        return query

    def clean_query(self, query):
        """
        Cleans and formats the generated query.

        Parameters:
        - query (str): The raw query string from the LLM.

        Returns:
        - query (str): The cleaned query string.
        """
        if not query:
            return None

        # Ensure query is a string
        if not isinstance(query, str):
            query = str(query)

        query = re.sub(r'^[{\["\'`]|[}\]"\'`]$', '', query)
        main_query_match = re.search(r'\(\((.*?)\)\)', query, re.DOTALL)
        if main_query_match:
            query = main_query_match.group(1)

        query = re.sub(
            r'\[?(\d{4}/\d{2}/\d{2})\s*:\s*(\d{4})\]?',
            r'("\1"[Date - Publication] : "\2"[Date - Publication])',
            query
        )
        query = f"(({query}))"
        query = re.sub(r'\s+', ' ', query).strip()

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

    def parse_json_safely(self, json_string):
        """
        Safely parses a JSON string, handling exceptions.

        Parameters:
        - json_string (str): The JSON string to parse.

        Returns:
        - dict: Parsed JSON data or empty dict if parsing fails.
        """
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}\nResponse was: {json_string}")
            return {}
#
# # Import the necessary modules
# from langchain_ollama.chat_models import ChatOllama
# import os
# from dotenv import load_dotenv
# import requests
# import json
#
# # Authenticate and get JWT token if required
# dotenv_path = '/Users/fernando/Documents/Research/academatepy/.env'
# load_dotenv(dotenv_path)
#
# auth_response = requests.post(os.getenv('AUTH_URL'), json= {'email': os.getenv('ollama_user'), 'password': os.getenv('ollama_pw')})
# jwt = json.loads(auth_response.text)["token"]
# headers = {"Authorization": "Bearer " + jwt}
#
# # Create an instance of the LLM
# ollama_llm = ChatOllama(
#     base_url=os.getenv('OLLAMA_API_URL'),
#     model='mistral:v0.2',
#     temperature=0.0,
#     client_kwargs={'headers': headers},
#     format="json"
# )
#
# # Create an instance of the PubMedQueryGenerator
# email = "your.email@example.com"
# api_key = os.getenv("NCBI_API_KEY")  # Replace with your actual NCBI API key
#
# pubmed_query_gen = PubMedQueryGenerator(email, api_key, ollama_llm)
#
# # Optionally set date filters
# pubmed_query_gen.set_date_filter(from_year=2010)
#
# # Generate a query for a topic
# topic = "The role of artificial intelligence in drug discovery"
# known_pmids = ["33844136", "33099022"]  # Optional known PMIDs to tailor the query
#
# query = pubmed_query_gen.generate_query(topic, known_pmids=known_pmids)
#
# print(f"Generated Query:\n{query}")
#
# # Execute the query
# pubmed_query_gen.execute_query(query)
#
# # Get the results as a DataFrame
# results_df = pubmed_query_gen.get_results_dataframe()
#
# print(results_df.head())
#
# models = ['reflection:70b', 'llama3:8b-instruct-fp16', 'llama3:latest', 'mistral:v0.2', 'mistral-nemo:latest', 'medllama2:latest', 'meditron:70b', 'meditron:7b', 'llama3.1:8b', 'llama3.1:70b', 'gemma:latest', 'phi3:latest', 'phi3:14b', 'gemma2:9b']
#
# for model_name in models:
#     print(f"Testing model: {model_name}")
#     # Create an instance of the LLM with the current model
#     ollama_llm = ChatOllama(
#         base_url=os.getenv('OLLAMA_API_URL'),
#         model=model_name,
#         temperature=0.0,
#         client_kwargs={'headers': headers},
#         format="json"
#     )
#     # Create a new instance of the PubMedQueryGenerator
#     pubmed_query_gen = PubMedQueryGenerator(email, api_key, ollama_llm)
#     pubmed_query_gen.set_date_filter(from_year=2010)
#     # Generate the query
#     query = pubmed_query_gen.generate_pubmed_query(topic)
#     print(f"Generated Query with {model_name}:\n{query}\n")
#     # Optionally, you can test executing the query and handle exceptions
#     try:
#         pubmed_query_gen.execute_query(query)
#         results_df = pubmed_query_gen.get_results_dataframe()
#         print(f"Results with {model_name}:\n{results_df.head()}\n")
#     except Exception as e:
#         print(f"Error with model {model_name}: {e}\n")
