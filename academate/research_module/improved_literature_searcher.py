"""
Improved unified literature searcher to find research articles across multiple databases.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Union
import time
import traceback

# Import query generator modules
from academate.research_module.improved_pubmed_querying import ImprovedPubMedQueryGenerator
from academate.research_module.improved_sscholar_querying import ImprovedSemanticScholarQueryGenerator
from academate.research_module.improved_keywords_generation import ImprovedKeywordsGenerator


class ImprovedLiteratureSearcher:
    """
    A unified searcher for finding research literature across multiple databases.

    This class coordinates searches across PubMed, Scopus, and Semantic Scholar,
    with unified result handling and duplicate removal.
    """

    def __init__(
            self,
            pubmed_email: Optional[str] = None,
            pubmed_api_key: Optional[str] = None,
            scopus_api_key: Optional[str] = None,
            semantic_scholar_api_key: Optional[str] = None,
            llm=None,
            from_year: Optional[int] = None,
            to_year: Optional[int] = None,
            max_results: int = 2000,
            two_step_search_queries: bool = True
    ):
        """
        Initialize the UnifiedLiteratureSearcher.

        Args:
            pubmed_email: Email address for PubMed API
            pubmed_api_key: NCBI API key for PubMed
            scopus_api_key: Elsevier Scopus API key
            semantic_scholar_api_key: Semantic Scholar API key
            llm: Language model instance for query generation
            from_year: Start year for date filtering
            to_year: End year for date filtering
            max_results: Maximum results per database
            two_step_search_queries: Whether to use two-step search approach
        """
        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self._setup_logger_handler()

        # Store configuration
        self.max_results = max_results
        self.two_step_search_queries = two_step_search_queries
        self.from_year = from_year
        self.to_year = to_year

        # Initialize query generators
        self.pubmed_query_gen = None
        self.scopus_query_gen = None
        self.semantic_scholar_query_gen = None
        self.keywords_gen = None

        # Store generated queries
        self.pubmed_query = None
        self.scopus_query = None
        self.semantic_scholar_query = None

        # Store results
        self.pubmed_results = pd.DataFrame()
        self.scopus_results = pd.DataFrame()
        self.semantic_scholar_results = pd.DataFrame()
        self.results = pd.DataFrame()

        # Set up PubMed query generator if credentials provided
        if pubmed_email and pubmed_api_key:
            self.pubmed_query_gen = ImprovedPubMedQueryGenerator(
                pubmed_email,
                pubmed_api_key,
                llm,
                two_step_search_queries
            )
            if from_year or to_year:
                self.pubmed_query_gen.set_date_filter(from_year=from_year, to_year=to_year)
        else:
            self.logger.warning("PubMed API credentials not provided. PubMed search disabled.")

        # Set up Scopus query generator if API key provided
        if scopus_api_key:
            # Replace with your improved Scopus query generator
            self.logger.warning("Improved Scopus query generator not implemented yet.")
            # self.scopus_query_gen = ImprovedScopusQueryGenerator(
            #     scopus_api_key,
            #     llm,
            #     two_step_search_queries
            # )
            # if from_year or to_year:
            #     self.scopus_query_gen.set_date_filter(from_year=from_year, to_year=to_year)
        else:
            self.logger.warning("Scopus API key not provided. Scopus search disabled.")

        # Set up Semantic Scholar query generator if API key provided
        if semantic_scholar_api_key:
            if semantic_scholar_api_key:
                self.semantic_scholar_query_gen = ImprovedSemanticScholarQueryGenerator(
                    semantic_scholar_api_key,
                    llm,
                    two_step_search_queries
                )
                if from_year or to_year:
                    self.semantic_scholar_query_gen.set_date_filter(from_year=from_year, to_year=to_year)
        else:
            self.logger.warning("Semantic Scholar API key not provided. Semantic Scholar search disabled.")

        # Set up Keywords generator if using two-step search
        if two_step_search_queries and llm:
            self.keywords_gen = ImprovedKeywordsGenerator(llm)

    def _setup_logger_handler(self):
        """Set up the logger handler if not already configured."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def search(self, topic: str):
        """
        Search for literature on the given topic across all configured databases.

        Args:
            topic: Research topic to search for

        Returns:
            None (results stored in self.results)
        """
        search_start_time = time.time()
        self.logger.info(f"Starting literature search for topic: {topic}")

        # Initialize empty dataframes for each database
        self.pubmed_results = pd.DataFrame()
        self.scopus_results = pd.DataFrame()
        self.semantic_scholar_results = pd.DataFrame()

        # Generate keywords if using two-step search
        if self.two_step_search_queries and self.keywords_gen:
            try:
                self.logger.info("Generating keywords...")
                keywords = self.keywords_gen.generate_keywords(topic)
                self.logger.info(f"Generated keywords: {keywords}")
                # Use generated keywords as topic
                search_topic = keywords
            except Exception as e:
                self.logger.error(f"Error generating keywords: {e}")
                self.logger.error(traceback.format_exc())
                # Fall back to original topic
                search_topic = topic
        else:
            search_topic = topic

        # Search PubMed if available
        if self.pubmed_query_gen:
            self._search_pubmed(search_topic)

        # Search Scopus if available
        if self.scopus_query_gen:
            self._search_scopus(search_topic)

        # Search Semantic Scholar if available
        if self.semantic_scholar_query_gen:
            self._search_semantic_scholar(search_topic)

        # Unify results if any are available
        if (not self.pubmed_results.empty or
                not self.scopus_results.empty or
                not self.semantic_scholar_results.empty):

            self.logger.info("Unifying results and removing duplicates...")
            self.results = self._unify_results(
                self.pubmed_results,
                self.scopus_results,
                self.semantic_scholar_results
            )
            self.logger.info(f"Found {len(self.results)} unique articles after deduplication.")
        else:
            self.logger.warning("No results found in any database.")
            self.results = pd.DataFrame()

        search_duration = time.time() - search_start_time
        self.logger.info(f"Literature search completed in {search_duration:.2f} seconds.")

    def _search_pubmed(self, topic: str):
        """Search PubMed database."""
        try:
            self.logger.info("Generating PubMed query...")
            pubmed_query = self.pubmed_query_gen.generate_query(topic)
            self.pubmed_query = pubmed_query  # Store query for reference

            if not pubmed_query:
                self.logger.error("Generated empty PubMed query.")
                return

            self.logger.info(f"PubMed Query: {pubmed_query}")

            # Execute query
            self.logger.info("Executing PubMed search...")
            self.pubmed_query_gen.execute_query(pubmed_query, max_results=self.max_results)
            self.pubmed_results = self.pubmed_query_gen.get_results_dataframe()

            if self.pubmed_results.empty:
                self.logger.warning("No results retrieved from PubMed.")
            else:
                self.logger.info(f"Retrieved {len(self.pubmed_results)} results from PubMed.")

        except Exception as e:
            self.logger.error(f"Error during PubMed search: {e}")
            self.logger.error(traceback.format_exc())
            self.pubmed_results = pd.DataFrame()

    def _search_scopus(self, topic: str):
        """Search Scopus database."""
        try:
            self.logger.info("Generating Scopus query...")
            scopus_query = self.scopus_query_gen.generate_query(topic)
            self.scopus_query = scopus_query  # Store query for reference

            if not scopus_query:
                self.logger.error("Generated empty Scopus query.")
                return

            self.logger.info(f"Scopus Query: {scopus_query}")

            # Execute query
            self.logger.info("Executing Scopus search...")
            self.scopus_query_gen.execute_query(scopus_query, max_results=self.max_results)
            self.scopus_results = self.scopus_query_gen.get_results_dataframe()

            if self.scopus_results.empty:
                self.logger.warning("No results retrieved from Scopus.")
            else:
                self.logger.info(f"Retrieved {len(self.scopus_results)} results from Scopus.")

        except Exception as e:
            self.logger.error(f"Error during Scopus search: {e}")
            self.logger.error(traceback.format_exc())
            self.scopus_results = pd.DataFrame()

    def _search_semantic_scholar(self, topic: str):
        """Search Semantic Scholar database."""
        try:
            self.logger.info("Generating Semantic Scholar query...")
            ss_query = self.semantic_scholar_query_gen.generate_query(topic)
            self.semantic_scholar_query = ss_query  # Store query for reference

            if not ss_query:
                self.logger.error("Generated empty Semantic Scholar query.")
                return

            self.logger.info(f"Semantic Scholar Query: {ss_query}")

            # Execute query
            self.logger.info("Executing Semantic Scholar search...")
            self.semantic_scholar_query_gen.execute_query(ss_query)
            self.semantic_scholar_results = self.semantic_scholar_query_gen.get_results_dataframe()

            if self.semantic_scholar_results.empty:
                self.logger.warning("No results retrieved from Semantic Scholar.")
            else:
                self.logger.info(f"Retrieved {len(self.semantic_scholar_results)} results from Semantic Scholar.")

        except Exception as e:
            self.logger.error(f"Error during Semantic Scholar search: {e}")
            self.logger.error(traceback.format_exc())
            self.semantic_scholar_results = pd.DataFrame()

    def _unify_results(self, pubmed_df: pd.DataFrame, scopus_df: pd.DataFrame,
                       semantic_scholar_df: pd.DataFrame) -> pd.DataFrame:
        """
        Unify results from different databases into a single DataFrame.

        Args:
            pubmed_df: PubMed results
            scopus_df: Scopus results
            semantic_scholar_df: Semantic Scholar results

        Returns:
            pd.DataFrame: Unified results with duplicates removed
        """
        dataframes = []

        # Process PubMed results
        if not pubmed_df.empty:
            try:
                pubmed_df = pubmed_df.rename(columns={
                    'PMID': 'id',
                    'Title': 'title',
                    'Abstract': 'abstract',
                    'DOI': 'doi'
                })
                pubmed_df['source'] = 'PubMed'
                dataframes.append(pubmed_df)
            except Exception as e:
                self.logger.error(f"Error processing PubMed results: {e}")

        # Process Scopus results
        if not scopus_df.empty:
            try:
                scopus_df = scopus_df.rename(columns={
                    'EID': 'id',
                    'Title': 'title',
                    'Abstract': 'abstract',
                    'DOI': 'doi'
                })
                scopus_df['source'] = 'Scopus'
                dataframes.append(scopus_df)
            except Exception as e:
                self.logger.error(f"Error processing Scopus results: {e}")

        # Process Semantic Scholar results
        if not semantic_scholar_df.empty:
            try:
                semantic_scholar_df = semantic_scholar_df.rename(columns={
                    'Paper ID': 'id',
                    'Title': 'title',
                    'Abstract': 'abstract',
                    'DOI': 'doi'
                })
                # Semantic Scholar data may not have DOI
                if 'doi' not in semantic_scholar_df.columns:
                    semantic_scholar_df['doi'] = pd.NA
                semantic_scholar_df['source'] = 'Semantic Scholar'
                dataframes.append(semantic_scholar_df)
            except Exception as e:
                self.logger.error(f"Error processing Semantic Scholar results: {e}")

        if not dataframes:
            self.logger.warning("No dataframes to unify.")
            return pd.DataFrame()

        # Combine dataframes
        try:
            combined_df = pd.concat(dataframes, ignore_index=True)

            # Handle missing DOIs
            combined_df['doi'] = combined_df['doi'].replace('', pd.NA)
            combined_df['doi'] = combined_df['doi'].fillna('missing_' + combined_df['id'].astype(str))

            # Group by DOI and aggregate sources
            combined_df['source'] = combined_df.groupby('doi')['source'].transform(lambda x: '; '.join(sorted(set(x))))
            combined_df = combined_df.drop_duplicates(subset='doi', keep='first')

            # Create Record column
            combined_df['Record'] = combined_df['title'].fillna('') + '\n' + combined_df['abstract'].fillna('')

            # Generate unique ID
            if 'uniqueid' not in combined_df.columns:
                combined_df['uniqueid'] = combined_df['doi'].apply(self._generate_uniqueid)

            return combined_df

        except Exception as e:
            self.logger.error(f"Error unifying results: {e}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _generate_uniqueid(self, text: str) -> str:
        """
        Generate a unique ID from text.

        Args:
            text: Input text

        Returns:
            str: Unique ID
        """
        import hashlib
        import re

        # Normalize text
        if not isinstance(text, str):
            text = str(text)

        # Remove special characters except basic punctuation
        normalized = re.sub(r'[^a-zA-Z0-9 \-():]', '', text)

        # Normalize whitespace
        normalized = ' '.join(normalized.split())

        # Generate hash
        return hashlib.sha256(normalized.encode()).hexdigest()[:20]

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get unified search results.

        Returns:
            pd.DataFrame: Unified results DataFrame
        """
        return self.results

    def get_queries(self) -> Dict[str, str]:
        """
        Get the queries used for each database.

        Returns:
            Dict[str, str]: Dictionary of queries by database
        """
        queries = {}
        if self.pubmed_query:
            queries['PubMed'] = self.pubmed_query
        if self.scopus_query:
            queries['Scopus'] = self.scopus_query
        if self.semantic_scholar_query:
            queries['Semantic Scholar'] = self.semantic_scholar_query
        return queries

    def get_database_results(self) -> Dict[str, pd.DataFrame]:
        """
        Get individual database results before unification.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of database results
        """
        results = {}
        if not self.pubmed_results.empty:
            results['PubMed'] = self.pubmed_results
        if not self.scopus_results.empty:
            results['Scopus'] = self.scopus_results
        if not self.semantic_scholar_results.empty:
            results['Semantic Scholar'] = self.semantic_scholar_results
        return results

    def save_results(self, filename: str) -> str:
        """
        Save results to a file.

        Args:
            filename: Output filename

        Returns:
            str: Path to saved file
        """
        if self.results.empty:
            self.logger.warning("No results to save.")
            return ""

        try:
            if filename.endswith('.csv'):
                self.results.to_csv(filename, index=False)
            elif filename.endswith('.xlsx'):
                self.results.to_excel(filename, index=False)
            elif filename.endswith('.json'):
                self.results.to_json(filename, orient='records', indent=2)
            else:
                # Default to CSV
                filename = filename + '.csv'
                self.results.to_csv(filename, index=False)

            self.logger.info(f"Results saved to {filename}")
            return filename

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return ""