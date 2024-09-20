import pandas as pd
import logging
from typing import List, Optional
from app.research_module.pubmed_querying import PubMedQueryGenerator
from app.research_module.scopus_querying import ScopusQueryGenerator


class UnifiedLiteratureSearcher:
    def __init__(self,
                 pubmed_email: Optional[str],
                 pubmed_api_key: Optional[str],
                 scopus_api_key: Optional[str],
                 llm,
                 from_year: Optional[int] = None,
                 to_year: Optional[int] = None):
        """
        Initializes the UnifiedLiteratureSearcher with PubMed and Scopus API keys and an LLM instance.

        Parameters:
        - pubmed_email (str): Email address for PubMed API usage.
        - pubmed_api_key (str): NCBI API key for PubMed.
        - scopus_api_key (str): Elsevier Scopus API key.
        - llm: An instance of the LLM for generating and improving queries.
        - from_year (int): Start year for date filtering (optional).
        - to_year (int): End year for date filtering (optional).
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.pubmed_query_gen = None
        self.scopus_query_gen = None

        if pubmed_email and pubmed_api_key:
            self.pubmed_query_gen = PubMedQueryGenerator(pubmed_email, pubmed_api_key, llm)
            if from_year or to_year:
                self.pubmed_query_gen.set_date_filter(from_year=from_year, to_year=to_year)
        else:
            self.logger.warning("PubMed API credentials not provided. PubMed access will not be available.")

        if scopus_api_key:
            self.scopus_query_gen = ScopusQueryGenerator(scopus_api_key, llm)
            if from_year or to_year:
                self.scopus_query_gen.set_date_filter(from_year=from_year, to_year=to_year)
        else:
            self.logger.warning("Scopus API key not provided. Scopus access will not be available.")

        self.from_year = from_year
        self.to_year = to_year
        self.results = None

    def search(self, topic: str, known_ids: Optional[List[str]] = None):
        """
        Searches PubMed and Scopus for the given topic and unifies the results.

        Parameters:
        - topic (str): The research topic to search for.
        - known_ids (list): Optional list of known IDs to tailor the query.
        """
        # Initialize empty dataframes
        pubmed_results = pd.DataFrame()
        scopus_results = pd.DataFrame()

        # Generate and execute PubMed query if access is available
        if self.pubmed_query_gen:
            try:
                self.logger.info("Generating PubMed query...")
                pubmed_query = self.pubmed_query_gen.generate_query(topic, known_pmids=known_ids)
                self.logger.info(f"PubMed Query:\n{pubmed_query}")
                self.logger.info("Executing PubMed search...")
                self.pubmed_query_gen.execute_query(pubmed_query)
                pubmed_results = self.pubmed_query_gen.get_results_dataframe()
                if pubmed_results is not None and not pubmed_results.empty:
                    self.logger.info(f"{len(pubmed_results)} results in PubMed.")
                else:
                    self.logger.warning("No results retrieved from PubMed.")
            except Exception as e:
                self.logger.error(f"Error during PubMed search: {e}")
        else:
            self.logger.warning("PubMed access not available.")

        # Generate and execute Scopus query if access is available
        if self.scopus_query_gen:
            try:
                self.logger.info("Generating Scopus query...")
                scopus_query = self.scopus_query_gen.generate_query(topic, known_ids=known_ids)
                self.logger.info(f"Scopus Query:\n{scopus_query}")
                self.logger.info("Executing Scopus search...")
                self.scopus_query_gen.execute_query(scopus_query)
                scopus_results = self.scopus_query_gen.get_results_dataframe()
                if scopus_results is not None and not scopus_results.empty:
                    self.logger.info(f"{len(scopus_results)} results in Scopus.")
                else:
                    self.logger.warning("No results retrieved from Scopus.")
            except Exception as e:
                self.logger.error(f"Error during Scopus search: {e}")
        else:
            self.logger.warning("Scopus access not available.")

        # Unify results if any are available
        if not pubmed_results.empty or not scopus_results.empty:
            self.logger.info("Unifying results and removing duplicates...")
            self.results = self.unify_results(pubmed_results, scopus_results)
            self.logger.info(f"{len(self.results)} results after deduplication.")
        else:
            self.logger.warning("No results to unify from any database.")
            self.results = pd.DataFrame()

    def unify_results(self, pubmed_df: pd.DataFrame, scopus_df: pd.DataFrame) -> pd.DataFrame:
        """
        Unifies PubMed and Scopus results into a single DataFrame, removing duplicates based on DOI.
        For articles found in multiple sources, combines the sources into a single entry.

        Parameters:
        - pubmed_df (pd.DataFrame): DataFrame containing PubMed results.
        - scopus_df (pd.DataFrame): DataFrame containing Scopus results.

        Returns:
        - pd.DataFrame: Unified DataFrame with duplicates removed and sources combined.
        """
        dataframes = []

        # Process PubMed results if available
        if pubmed_df is not None and not pubmed_df.empty:
            pubmed_df = pubmed_df.rename(columns={
                'PMID': 'id',
                'Title': 'title',
                'Abstract': 'abstract',
                'DOI': 'doi'
            })
            pubmed_df['source'] = 'PubMed'
            dataframes.append(pubmed_df)
        else:
            self.logger.warning("PubMed results are empty or not available.")

        # Process Scopus results if available
        if scopus_df is not None and not scopus_df.empty:
            scopus_df = scopus_df.rename(columns={
                'EID': 'id',
                'Title': 'title',
                'Abstract': 'abstract',
                'DOI': 'doi'
            })
            scopus_df['source'] = 'Scopus'
            dataframes.append(scopus_df)
        else:
            self.logger.warning("Scopus results are empty or not available.")

        if not dataframes:
            self.logger.error("No dataframes to unify. Returning empty DataFrame.")
            return pd.DataFrame()

        # Combine the dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Handle missing DOIs by filling with a unique identifier
        combined_df['doi'] = combined_df['doi'].replace('', pd.NA)
        combined_df['doi'] = combined_df['doi'].fillna('missing_' + combined_df['id'].astype(str))

        # Group by DOI and aggregate sources
        combined_df['source'] = combined_df.groupby('doi')['source'].transform(lambda x: '; '.join(sorted(set(x))))
        combined_df = combined_df.drop_duplicates(subset='doi', keep='first')

        return combined_df

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Returns the unified search results as a pandas DataFrame.

        Returns:
        - pd.DataFrame: DataFrame containing the unified results.
        """
        return self.results


from langchain_ollama.chat_models import ChatOllama
import os
from dotenv import load_dotenv
import requests
import json
import pybliometrics.scopus
pybliometrics.scopus.init()

# Authenticate and get JWT token if required
dotenv_path = '/Users/fernando/Documents/Research/academatepy/.env'
load_dotenv(dotenv_path)

auth_response = requests.post(os.getenv('AUTH_URL'), json= {'email': os.getenv('ollama_user'), 'password': os.getenv('ollama_pw')})
jwt = json.loads(auth_response.text)["token"]
headers = {"Authorization": "Bearer " + jwt}

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
# # Your API keys
# pubmed_email = 'your.email@example.com'
# pubmed_api_key = os.getenv('NCBI_API_KEY')
# scopus_api_key = os.getenv('SCOPUS_API_KEY')
#
#
# # Create an instance of the UnifiedLiteratureSearcher
# searcher = UnifiedLiteratureSearcher(pubmed_email, pubmed_api_key, scopus_api_key, llm=ollama_llm, from_year=2010, to_year=2023)
# self = searcher
# # Perform the search
# topic = "The role of artificial intelligence in drug discovery"
# known_ids = ["33844136", "33099022"]  # Example known PMIDs for PubMed
#
# searcher.search(topic, known_ids=known_ids)
#
# # Get the unified results
# results_df = searcher.get_results_dataframe()
#
# # Display articles with their DOIs
# print(results_df[['title', 'doi', 'source']].head())
#
# # Get the unified results
# results_df = searcher.get_results_dataframe()
#
# # Verify the presence of the 'source' column
# print("Columns in results_df:")
# print(results_df.columns)
#
# # Display the results with the 'source' column
# print("\nFirst few articles with source information:")
# print(results_df[['title', 'source']].head())
#
# # Optionally, display all columns
# pd.set_option('display.max_columns', None)
# print("\nFull DataFrame:")
# print(results_df.head())
#
# # Analyze the number of articles from each source
# print("\nNumber of articles from each source:")
# print(results_df['source'].value_counts())


models = ['gemma2:9b',
    'llama3:8b-instruct-fp16',
    'llama3:latest',
    'mistral:v0.2',
    'mistral-nemo:latest',
    'medllama2:latest',
    'meditron:70b',
    'meditron:7b',
    'llama3.1:8b',
    'llama3.1:70b',
    'gemma:latest',
    'phi3:latest',
    'phi3:14b',
    'reflection:70b',

]

# DataFrame to store results
results_summary = pd.DataFrame(columns=[
    'Model',
    'Valid PubMed Query',
    'PubMed Articles Retrieved',
    'Valid Scopus Query',
    'Scopus Articles Retrieved',
    'Total Articles Retrieved'
])

# Import necessary modules
import logging
from langchain_ollama.chat_models import ChatOllama
import os
from dotenv import load_dotenv
import requests
import json
import pybliometrics.scopus

pybliometrics.scopus.init()

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

# Your API keys
pubmed_email = 'your.email@example.com'
pubmed_api_key = os.getenv('NCBI_API_KEY')
scopus_api_key = os.getenv('SCOPUS_API_KEY')

# The research topic
topic = "The role of artificial intelligence in drug discovery"
known_ids = ["33844136", "33099022"]  # Example known PMIDs for PubMed


# DataFrame to store results
results_summary = pd.DataFrame(columns=[
    'Model',
    'Valid PubMed Query',
    'PubMed Articles Retrieved',
    'Valid Scopus Query',
    'Scopus Articles Retrieved',
    'Total Articles Retrieved'
])

# Loop through the list of models
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

    # Create an instance of the UnifiedLiteratureSearcher
    searcher = UnifiedLiteratureSearcher(
        pubmed_email,
        pubmed_api_key,
        scopus_api_key,
        llm=ollama_llm,
        from_year=2010,
        to_year=2023
    )

    # Initialize variables to track validity and article counts
    valid_pubmed_query = False
    pubmed_articles_retrieved = 0
    valid_scopus_query = False
    scopus_articles_retrieved = 0

    # Perform the search
    try:
        searcher.search(topic, known_ids=known_ids)
        # Get the results
        results_df = searcher.get_results_dataframe()

        # Check if PubMed query was valid
        if searcher.pubmed_query_gen:
            pubmed_query = searcher.pubmed_query_gen.last_query  # Assuming you store the last query
            if pubmed_query:
                valid_pubmed_query = True
                pubmed_articles_retrieved = len(searcher.pubmed_query_gen.get_results_dataframe())
            else:
                valid_pubmed_query = False

        # Check if Scopus query was valid
        if searcher.scopus_query_gen:
            scopus_query = searcher.scopus_query_gen.last_query  # Assuming you store the last query
            if scopus_query:
                valid_scopus_query = True
                scopus_articles_retrieved = len(searcher.scopus_query_gen.get_results_dataframe())
            else:
                valid_scopus_query = False

        total_articles_retrieved = len(results_df)

        # Add the results to the summary DataFrame using pd.concat
        new_row = {
            'Model': model_name,
            'Valid PubMed Query': valid_pubmed_query,
            'PubMed Articles Retrieved': pubmed_articles_retrieved,
            'Valid Scopus Query': valid_scopus_query,
            'Scopus Articles Retrieved': scopus_articles_retrieved,
            'Total Articles Retrieved': total_articles_retrieved
        }
        results_summary = pd.concat([results_summary, pd.DataFrame([new_row])], ignore_index=True)

        print(f"Model {model_name} retrieved {total_articles_retrieved} articles.")


    except Exception as e:
        print(f"Error with model {model_name}: {e}")
        # In case of error, add the model with zero articles retrieved
        new_row = {
            'Model': model_name,
            'Valid PubMed Query': valid_pubmed_query,
            'PubMed Articles Retrieved': pubmed_articles_retrieved,
            'Valid Scopus Query': valid_scopus_query,
            'Scopus Articles Retrieved': scopus_articles_retrieved,
            'Total Articles Retrieved': 0
        }
        results_summary = pd.concat([results_summary, pd.DataFrame([new_row])], ignore_index=True)

# Display the summary results
print("\nSummary of LLM Performance:")
print(results_summary)

# Sort by Total Articles Retrieved
sorted_results = results_summary.sort_values(by='Total Articles Retrieved', ascending=False)
print("\nLLMs sorted by Total Articles Retrieved:")
print(sorted_results[['Model', 'Total Articles Retrieved']])

# Identify LLMs that generated valid queries for both databases
valid_both = results_summary[
    (results_summary['Valid PubMed Query'] == True) &
    (results_summary['Valid Scopus Query'] == True)
]
print("\nLLMs that generated valid queries for both PubMed and Scopus:")
print(valid_both['Model'].tolist())
