import pandas as pd
import logging
from typing import List, Optional

import sys
sys.path.append("/Users/fernando/Documents/Research/academatepy/app/research_module")
from pubmed_querying import PubMedQueryGenerator
from scopus_querying import ScopusQueryGenerator
from sscholar_querying import SemanticScholarQueryGenerator


class UnifiedLiteratureSearcher:
    def __init__(self,
                 pubmed_email: Optional[str],
                 pubmed_api_key: Optional[str],
                 scopus_api_key: Optional[str],
                 semantic_scholar_api_key: Optional[str],
                 llm,
                 from_year: Optional[int] = None,
                 to_year: Optional[int] = None,
                 scopus_max_results: int = 2000):
        """
        Initializes the UnifiedLiteratureSearcher with PubMed, Scopus, and Semantic Scholar API keys and an LLM instance.

        Parameters:
        - pubmed_email (str): Email address for PubMed API usage.
        - pubmed_api_key (str): NCBI API key for PubMed.
        - scopus_api_key (str): Elsevier Scopus API key.
        - semantic_scholar_api_key (str): Semantic Scholar API key.
        - llm: An instance of the LLM for generating and improving queries.
        - from_year (int): Start year for date filtering (optional).
        - to_year (int): End year for date filtering (optional).
        - scopus_max_results (int): Maximum number of Scopus results to fetch.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.pubmed_query_gen = None
        self.scopus_query_gen = None
        self.semantic_scholar_query_gen = None

        self.pubmed_query = None
        self.scopus_query = None
        self.semantic_scholar_query = None

        self.pubmed_results = pd.DataFrame()
        self.scopus_results = pd.DataFrame()
        self.semantic_scholar_results = pd.DataFrame()

        self.scopus_max_results = scopus_max_results

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

        if semantic_scholar_api_key:
            self.semantic_scholar_query_gen = SemanticScholarQueryGenerator(semantic_scholar_api_key, llm)
            if from_year or to_year:
                self.semantic_scholar_query_gen.set_date_filter(from_year=from_year, to_year=to_year)
        else:
            self.logger.warning("Semantic Scholar API key not provided. Semantic Scholar access will not be available.")

        self.from_year = from_year
        self.to_year = to_year
        self.results = None

    def search(self, topic: str):
        """
        Searches PubMed, Scopus, and Semantic Scholar for the given topic and unifies the results.

        Parameters:
        - topic (str): The research topic to search for.
        """
        # Initialize empty dataframes
        self.pubmed_results = pd.DataFrame()
        self.scopus_results = pd.DataFrame()
        self.semantic_scholar_results = pd.DataFrame()

        # Generate and execute PubMed query if access is available
        if self.pubmed_query_gen:
            try:
                self.logger.info("Generating PubMed query...")
                pubmed_query = self.pubmed_query_gen.generate_query(topic)
                self.pubmed_query = pubmed_query  # Store the query for traceability
                self.logger.info(f"PubMed Query:\n{pubmed_query}")
                self.logger.info("Executing PubMed search...")
                self.pubmed_query_gen.execute_query(pubmed_query)
                self.pubmed_results = self.pubmed_query_gen.get_results_dataframe()
                if self.pubmed_results is not None and not self.pubmed_results.empty:
                    self.logger.info(f"{len(self.pubmed_results)} results in PubMed.")
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
                scopus_query = self.scopus_query_gen.generate_query(topic)
                self.scopus_query = scopus_query  # Store the query for traceability
                self.logger.info(f"Scopus Query:\n{scopus_query}")
                self.logger.info("Executing Scopus search...")
                self.scopus_query_gen.execute_query(scopus_query, max_results=self.scopus_max_results)
                self.scopus_results = self.scopus_query_gen.get_results_dataframe()
                if self.scopus_results is not None and not self.scopus_results.empty:
                    self.logger.info(f"{len(self.scopus_results)} results in Scopus.")
                else:
                    self.logger.warning("No results retrieved from Scopus.")
            except Exception as e:
                self.logger.error(f"Error during Scopus search: {e}")
        else:
            self.logger.warning("Scopus access not available.")

        # Generate and execute Semantic Scholar query if access is available
        if self.semantic_scholar_query_gen:
            try:
                self.logger.info("Generating Semantic Scholar query...")
                semantic_scholar_query = self.semantic_scholar_query_gen.generate_query(topic)
                self.semantic_scholar_query = semantic_scholar_query  # Store the query for traceability
                self.logger.info(f"Semantic Scholar Query:\n{semantic_scholar_query}")
                self.logger.info("Executing Semantic Scholar search...")
                self.semantic_scholar_query_gen.execute_query(semantic_scholar_query)
                self.semantic_scholar_results = self.semantic_scholar_query_gen.get_results_dataframe()
                if self.semantic_scholar_results is not None and not self.semantic_scholar_results.empty:
                    self.logger.info(f"{len(self.semantic_scholar_results)} results in Semantic Scholar.")
                else:
                    self.logger.warning("No results retrieved from Semantic Scholar.")
            except Exception as e:
                self.logger.error(f"Error during Semantic Scholar search: {e}")
        else:
            self.logger.warning("Semantic Scholar access not available.")

        # Unify results if any are available
        if not self.pubmed_results.empty or not self.scopus_results.empty or not self.semantic_scholar_results.empty:
            self.logger.info("Unifying results and removing duplicates...")
            self.results = self.unify_results(self.pubmed_results, self.scopus_results, self.semantic_scholar_results)
            self.logger.info(f"{len(self.results)} results after deduplication.")
        else:
            self.logger.warning("No results to unify from any database.")
            self.results = pd.DataFrame()

    def unify_results(self, pubmed_df: pd.DataFrame, scopus_df: pd.DataFrame, semantic_scholar_df: pd.DataFrame) -> pd.DataFrame:
        """
        Unifies PubMed, Scopus, and Semantic Scholar results into a single DataFrame, removing duplicates based on DOI.
        For articles found in multiple sources, combines the sources into a single entry.

        Parameters:
        - pubmed_df (pd.DataFrame): DataFrame containing PubMed results.
        - scopus_df (pd.DataFrame): DataFrame containing Scopus results.
        - semantic_scholar_df (pd.DataFrame): DataFrame containing Semantic Scholar results.

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

        # Process Semantic Scholar results if available
        if semantic_scholar_df is not None and not semantic_scholar_df.empty:
            semantic_scholar_df = semantic_scholar_df.rename(columns={
                'Paper ID': 'id',
                'Title': 'title',
                'Abstract': 'abstract',
                'DOI': 'doi'
            })
            # Semantic Scholar data may not have DOI, need to handle it
            if 'doi' not in semantic_scholar_df.columns:
                semantic_scholar_df['doi'] = pd.NA  # Fill with NaN
            semantic_scholar_df['source'] = 'Semantic Scholar'
            dataframes.append(semantic_scholar_df)
        else:
            self.logger.warning("Semantic Scholar results are empty or not available.")

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

        # Create the 'Record' column combining Title and Abstract
        combined_df['Record'] = combined_df['title'].fillna('') + '\n' + combined_df['abstract'].fillna('')

        return combined_df

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Returns the unified search results as a pandas DataFrame.

        Returns:
        - pd.DataFrame: DataFrame containing the unified results.
        """
        return self.results

    def get_queries(self):
        """
        Returns a dictionary of the queries used for each database.

        Returns:
        - dict: Dictionary containing the queries used.
        """
        queries = {}
        if self.pubmed_query:
            queries['PubMed'] = self.pubmed_query
        if self.scopus_query:
            queries['Scopus'] = self.scopus_query
        if self.semantic_scholar_query:
            queries['Semantic Scholar'] = self.semantic_scholar_query
        return queries

    def get_database_results(self):
        """
        Returns a dictionary containing the results from each database before unification.

        Returns:
        - dict: Dictionary containing the results DataFrames from each database.
        """
        results = {}
        if not self.pubmed_results.empty:
            results['PubMed'] = self.pubmed_results
        if not self.scopus_results.empty:
            results['Scopus'] = self.scopus_results
        if not self.semantic_scholar_results.empty:
            results['Semantic Scholar'] = self.semantic_scholar_results
        return results
