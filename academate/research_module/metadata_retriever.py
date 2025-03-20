import os
import re
import time
import pandas as pd
import numpy as np
import logging
import requests
import json
from tqdm import tqdm
from habanero import Crossref
from Bio import Entrez
from typing import Optional, Dict, List, Tuple, Union
import hashlib


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("article_retrieval.log"),
        logging.StreamHandler()
    ]
)


class ArticleMetadataRetriever:
    """
    A comprehensive tool to retrieve metadata (title, abstract, etc.) for academic articles
    using multiple fallback methods and sources.
    """

    def __init__(self, email: str, api_key: Optional[str] = None, verbose: bool = False):
        """
        Initialize the ArticleMetadataRetriever.

        Args:
            email (str): Email for API authentication (required by PubMed/Entrez)
            api_key (str, optional): API key for services that require it
            verbose (bool): Whether to show detailed logging
        """
        self.email = email
        self.api_key = api_key
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        # Configure services that require authentication
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key

        # Initialize Crossref client
        self.cr = Crossref(mailto=email)

        # Rate limiting helpers
        self.last_request_time = {}

        # Error tracking
        self.errors = {
            'doi_search': [],
            'metadata_retrieval': []
        }

        # Success counters
        self.successful_retrievals = 0
        self.failed_retrievals = 0

        # Cache to avoid duplicate API calls
        self.doi_cache = {}
        self.metadata_cache = {}

    @staticmethod
    def is_valid_doi(doi: str) -> bool:
        """Check if a DOI string matches the expected format."""
        if not doi or not isinstance(doi, str):
            return False
        doi_regex = re.compile(r'^10.\d{4,9}/[-._;()/:A-Z0-9]+$', re.IGNORECASE)
        return bool(doi_regex.match(doi))

    def _wait_for_rate_limit(self, service: str, min_wait: float = 0.1):
        """Manage rate limiting for API calls."""
        current_time = time.time()
        if service in self.last_request_time:
            elapsed = current_time - self.last_request_time[service]
            if elapsed < min_wait:
                time.sleep(min_wait - elapsed)

        self.last_request_time[service] = time.time()

    def find_dois(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find DOIs for articles in a DataFrame using multiple strategies with improved validation.

        Args:
            df (pd.DataFrame): DataFrame with article information

        Returns:
            pd.DataFrame: Updated DataFrame with DOI information
        """
        working_df = df.copy()

        # Initialize DOI columns if they don't exist
        if 'doi' not in working_df.columns:
            working_df['doi'] = None
        if 'doi_source' not in working_df.columns:
            working_df['doi_source'] = None

        # Create a column to track potential DOI conflicts
        working_df['_doi_conflict'] = False

        # Track DOI assignments to detect duplicates
        doi_assignments = {}

        # 1. Direct extraction: Look for DOI patterns in text columns with better validation
        text_columns = []
        for col in working_df.columns:
            # Only use text columns that are likely to contain unique content
            if col.lower() not in ['doi', 'uniqueid', 'pdf_path', 'doi_source'] and working_df[col].dtype == 'object':
                # Check if column likely contains unique content per row
                unique_values = working_df[col].nunique()
                total_values = working_df[col].notna().sum()

                # If more than 50% of values are unique, consider it a good content column
                if unique_values > 0 and total_values > 0 and unique_values / total_values > 0.5:
                    text_columns.append(col)

        self.logger.info(f"Using these columns for DOI extraction: {text_columns}")

        # Now search for DOIs in the selected text columns
        for col in text_columns:
            for idx, row in working_df.iterrows():
                if pd.isna(working_df.at[idx, 'doi']) or working_df.at[idx, 'doi'] == 'DOI not found':
                    if pd.notna(row[col]) and isinstance(row[col], str):
                        # Only consider DOIs that appear at start of text or are preceded by "doi:"
                        content = row[col].lower()

                        # Try direct DOI pattern (with validation to avoid false matches)
                        direct_matches = re.findall(r'(?:^|\s|doi:|\()(\b10\.\d{4,9}/[-._;()/:A-Z0-9]+)(?:\s|\)|$)',
                                                    content,
                                                    re.IGNORECASE)

                        if direct_matches:
                            # Pick the first valid match
                            for match in direct_matches:
                                potential_doi = match.strip()
                                if self.is_valid_doi(potential_doi):
                                    # Check if this DOI has been assigned to other rows
                                    if potential_doi in doi_assignments:
                                        # If this is a different row, flag potential conflict
                                        if doi_assignments[potential_doi] != idx:
                                            self.logger.warning(
                                                f"DOI conflict: {potential_doi} found in rows {doi_assignments[potential_doi]} and {idx}")
                                            working_df.at[idx, '_doi_conflict'] = True
                                    else:
                                        doi_assignments[potential_doi] = idx

                                    working_df.at[idx, 'doi'] = potential_doi
                                    working_df.at[idx, 'doi_source'] = f"{col}_direct"

                                    if self.verbose:
                                        self.logger.info(f"Found DOI via direct match in '{col}': {potential_doi}")
                                    break

        # 2. Use Crossref API with improved query building
        records_to_process = working_df[
            (working_df['doi'].isna() | (working_df['doi'] == 'DOI not found')) &
            (~working_df['_doi_conflict'])  # Skip conflicted records
            ]

        if len(records_to_process) > 0:
            self.logger.info(f"Searching for DOIs using Crossref API for {len(records_to_process)} records")

            # Progress bar
            tqdm.pandas(desc="Finding DOIs with Crossref")

            def get_doi_from_crossref(row):
                # Create a unique query key based on row content
                uniqueid = str(row.get('uniqueid', '')) if 'uniqueid' in row else ''

                # Build better query focusing on title-like fields first
                query_parts = []
                title_fields = ['title', 'Title', 'reference', 'Reference']

                # Try to use title fields first
                for field in title_fields:
                    if field in row and pd.notna(row[field]) and len(str(row[field]).strip()) > 20:
                        query_parts.append(str(row[field]))

                # If no good title fields, use other text content
                if not query_parts:
                    for col in text_columns:
                        if col in row and pd.notna(row[col]) and len(str(row[col]).strip()) > 20:
                            # For larger text fields, just use the first 200 chars
                            content = str(row[col]).strip()
                            if len(content) > 200:
                                content = content[:200]
                            query_parts.append(content)

                query = ' '.join(query_parts).strip()
                if not query:
                    return pd.Series(['DOI not found', None])

                # Create cache key using uniqueid + query to avoid conflicts
                cache_key = f"{uniqueid}_{query[:100]}"

                # Check cache to avoid duplicate API calls
                if cache_key in self.doi_cache:
                    return pd.Series(self.doi_cache[cache_key])

                # Search Crossref with multiple attempts
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        self._wait_for_rate_limit('crossref', 1.0)

                        # Use a more focused query
                        result = self.cr.works(query=query[:500], limit=5)  # Get top 5 results
                        items = result['message']['items']

                        if items:
                            # Try to match to the best result with validation
                            for item in items[:3]:  # Check top 3 results
                                doi = item.get('DOI', '')
                                if self.is_valid_doi(doi):
                                    # Basic validation - check if title somewhat matches (if we have titles)
                                    title_score = 0
                                    if 'title' in item and item['title'] and any(f in row for f in title_fields):
                                        item_title = item['title'][0].lower()
                                        for field in title_fields:
                                            if field in row and pd.notna(row[field]):
                                                row_title = str(row[field]).lower()
                                                # Simple overlap score based on common words
                                                item_words = set(re.findall(r'\b\w+\b', item_title))
                                                row_words = set(re.findall(r'\b\w+\b', row_title))
                                                if item_words and row_words:
                                                    overlap = len(item_words.intersection(row_words))
                                                    title_score = overlap / max(len(item_words), len(row_words))

                                    # Accept if good title match (>30% overlap) or no title to compare
                                    if title_score > 0.3 or title_score == 0:
                                        self.doi_cache[cache_key] = [doi, 'crossref']

                                        # Check for conflicts
                                        if doi in doi_assignments and doi_assignments[doi] != row.name:
                                            self.logger.warning(
                                                f"API DOI conflict: {doi} assigned to rows {doi_assignments[doi]} and {row.name}")
                                            # Return the DOI but mark it as conflicted
                                            return pd.Series([doi, 'crossref_conflict'])

                                        # Track this assignment
                                        doi_assignments[doi] = row.name
                                        return pd.Series([doi, 'crossref'])

                        # If we got a response but no valid DOI matches
                        if attempt == max_attempts - 1:
                            self.doi_cache[cache_key] = ['DOI not found', None]
                            return pd.Series(['DOI not found', None])

                    except Exception as e:
                        self.logger.warning(f"DOI search attempt {attempt + 1} failed: {str(e)[:100]}...")
                        self.errors['doi_search'].append({
                            'uniqueid': uniqueid,
                            'query': query[:100],
                            'error': str(e),
                            'attempt': attempt + 1
                        })
                        time.sleep(2 ** attempt)  # Exponential backoff

                self.doi_cache[cache_key] = ['DOI not found', None]
                return pd.Series(['DOI not found', None])

            # Apply the function to find DOIs
            try:
                records_to_process[['doi', 'doi_source']] = records_to_process.progress_apply(
                    get_doi_from_crossref, axis=1, result_type="expand"
                )

                # Update the main DataFrame
                working_df.update(records_to_process[['doi', 'doi_source']])

                # Mark conflicts
                conflict_mask = records_to_process['doi_source'] == 'crossref_conflict'
                if conflict_mask.any():
                    conflict_indices = records_to_process[conflict_mask].index
                    working_df.loc[conflict_indices, '_doi_conflict'] = True

            except Exception as e:
                self.logger.error(f"Error during DOI search with Crossref: {str(e)}")

        # 3. Final validation to remove obvious duplicates
        # Count how many times each DOI appears
        doi_counts = working_df['doi'].value_counts()
        duplicated_dois = doi_counts[doi_counts > 1].index.tolist()

        if duplicated_dois:
            self.logger.warning(f"Found {len(duplicated_dois)} DOIs assigned to multiple records.")
            for doi in duplicated_dois:
                # Get all rows with this DOI
                dup_rows = working_df[working_df['doi'] == doi]

                if len(dup_rows) > 1:
                    self.logger.warning(f"DOI {doi} is assigned to {len(dup_rows)} rows")

                    # Mark all as conflicts except the first one with direct match
                    direct_matches = dup_rows[dup_rows['doi_source'].str.contains('direct', na=False)]

                    if len(direct_matches) == 1:
                        # Keep only the direct match
                        to_clear = dup_rows.index.difference(direct_matches.index)
                        working_df.loc[to_clear, 'doi'] = None
                        working_df.loc[to_clear, 'doi_source'] = None
                    elif len(direct_matches) > 1:
                        # Multiple direct matches - keep all but mark as conflicts
                        working_df.loc[direct_matches.index, '_doi_conflict'] = True
                    else:
                        # No direct matches - keep the first and clear others
                        to_keep = dup_rows.index[0]
                        to_clear = dup_rows.index[1:]
                        working_df.loc[to_clear, 'doi'] = None
                        working_df.loc[to_clear, 'doi_source'] = None

        # Report results
        valid_doi_count = working_df['doi'].apply(lambda x: self.is_valid_doi(x) if pd.notna(x) else False).sum()
        conflict_count = working_df['_doi_conflict'].sum()

        self.logger.info(f"DOI search complete:")
        self.logger.info(f"- Found {valid_doi_count} valid DOIs out of {len(working_df)} records")
        self.logger.info(f"- Detected {conflict_count} potential DOI conflicts")

        # Remove temporary column
        working_df.drop('_doi_conflict', axis=1, inplace=True)

        return working_df

    def get_pubmed_metadata(self, doi: str) -> Dict:
        """
        Retrieve metadata from PubMed using a DOI.

        Args:
            doi (str): DOI of the article

        Returns:
            Dict: Dictionary with metadata (pmid, title, abstract, etc.)
        """
        if doi in self.metadata_cache:
            return self.metadata_cache[doi].get('pubmed', {})

        metadata = {
            'pmid': None,
            'title': None,
            'abstract': None,
            'journal': None,
            'authors': None,
            'publication_date': None,
            'success': False
        }

        try:
            self._wait_for_rate_limit('pubmed', 0.5)

            # First, search for the PubMed ID using the DOI
            handle = Entrez.esearch(db="pubmed", term=f"{doi}[DOI]", retmax=1)
            search_results = Entrez.read(handle)
            handle.close()

            if search_results["IdList"]:
                pmid = search_results["IdList"][0]
                metadata['pmid'] = pmid

                # Fetch the full record
                handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="text")
                record = Entrez.read(handle)
                handle.close()

                if record and 'PubmedArticle' in record and record['PubmedArticle']:
                    article = record['PubmedArticle'][0]

                    # Extract article data
                    if 'MedlineCitation' in article and 'Article' in article['MedlineCitation']:
                        article_data = article['MedlineCitation']['Article']

                        # Get title
                        if 'ArticleTitle' in article_data:
                            metadata['title'] = article_data['ArticleTitle']

                        # Get abstract
                        if 'Abstract' in article_data and 'AbstractText' in article_data['Abstract']:
                            abstract_sections = article_data['Abstract']['AbstractText']
                            abstract_text = []

                            for section in abstract_sections:
                                # Handle labeled sections
                                if hasattr(section, 'attributes') and 'Label' in section.attributes:
                                    label = section.attributes['Label']
                                    abstract_text.append(f"{label}: {section}")
                                else:
                                    abstract_text.append(str(section))

                            metadata['abstract'] = ' '.join(abstract_text)

                        # Get journal
                        if 'Journal' in article_data and 'Title' in article_data['Journal']:
                            metadata['journal'] = article_data['Journal']['Title']

                        # Get authors
                        if 'AuthorList' in article_data:
                            author_list = []
                            for author in article_data['AuthorList']:
                                if 'LastName' in author and 'ForeName' in author:
                                    author_list.append(f"{author['LastName']} {author['ForeName']}")
                                elif 'LastName' in author:
                                    author_list.append(author['LastName'])
                                elif 'CollectiveName' in author:
                                    author_list.append(author['CollectiveName'])

                            metadata['authors'] = ', '.join(author_list)

                        # Get publication date
                        if 'Journal' in article_data and 'JournalIssue' in article_data['Journal'] and 'PubDate' in \
                                article_data['Journal']['JournalIssue']:
                            pub_date = article_data['Journal']['JournalIssue']['PubDate']
                            date_parts = []

                            if 'Year' in pub_date:
                                date_parts.append(pub_date['Year'])
                            if 'Month' in pub_date:
                                date_parts.append(pub_date['Month'])
                            if 'Day' in pub_date:
                                date_parts.append(pub_date['Day'])

                            metadata['publication_date'] = ' '.join(date_parts)

                metadata['success'] = bool(metadata['title'] or metadata['abstract'])

                # Cache the result
                if doi not in self.metadata_cache:
                    self.metadata_cache[doi] = {}

                self.metadata_cache[doi]['pubmed'] = metadata

        except Exception as e:
            self.logger.warning(f"Error retrieving PubMed metadata for DOI {doi}: {str(e)[:100]}...")
            self.errors['metadata_retrieval'].append({
                'doi': doi,
                'source': 'pubmed',
                'error': str(e)
            })

        return metadata

    def get_crossref_metadata(self, doi: str) -> Dict:
        """
        Retrieve metadata from Crossref using a DOI.

        Args:
            doi (str): DOI of the article

        Returns:
            Dict: Dictionary with metadata
        """
        if doi in self.metadata_cache and 'crossref' in self.metadata_cache[doi]:
            return self.metadata_cache[doi]['crossref']

        metadata = {
            'title': None,
            'abstract': None,
            'journal': None,
            'authors': None,
            'publication_date': None,
            'success': False
        }

        try:
            self._wait_for_rate_limit('crossref', 1.0)

            # Fetch works by DOI
            result = self.cr.works(ids=doi)

            if result and 'message' in result:
                message = result['message']

                # Get title
                if 'title' in message and message['title']:
                    metadata['title'] = message['title'][0]

                # Get abstract
                if 'abstract' in message and message['abstract']:
                    # Clean HTML tags if present
                    abstract = message['abstract']
                    abstract = re.sub(r'<[^>]+>', '', abstract)
                    metadata['abstract'] = abstract

                # Get journal
                if 'container-title' in message and message['container-title']:
                    metadata['journal'] = message['container-title'][0]

                # Get authors
                if 'author' in message and message['author']:
                    author_list = []
                    for author in message['author']:
                        if 'family' in author and 'given' in author:
                            author_list.append(f"{author['family']} {author['given']}")
                        elif 'family' in author:
                            author_list.append(author['family'])

                    metadata['authors'] = ', '.join(author_list)

                # Get publication date
                if 'published' in message and 'date-parts' in message['published'] and message['published'][
                    'date-parts']:
                    date_parts = message['published']['date-parts'][0]
                    metadata['publication_date'] = '-'.join(str(part) for part in date_parts)

                metadata['success'] = bool(metadata['title'] or metadata['abstract'])

                # Cache the result
                if doi not in self.metadata_cache:
                    self.metadata_cache[doi] = {}

                self.metadata_cache[doi]['crossref'] = metadata

        except Exception as e:
            self.logger.warning(f"Error retrieving Crossref metadata for DOI {doi}: {str(e)[:100]}...")
            self.errors['metadata_retrieval'].append({
                'doi': doi,
                'source': 'crossref',
                'error': str(e)
            })

        return metadata

    def get_semanticscholar_metadata(self, doi: str) -> Dict:
        """
        Retrieve metadata from Semantic Scholar using a DOI.

        Args:
            doi (str): DOI of the article

        Returns:
            Dict: Dictionary with metadata
        """
        if doi in self.metadata_cache and 'semanticscholar' in self.metadata_cache[doi]:
            return self.metadata_cache[doi]['semanticscholar']

        metadata = {
            'title': None,
            'abstract': None,
            'journal': None,
            'authors': None,
            'publication_date': None,
            'success': False
        }

        try:
            self._wait_for_rate_limit('semanticscholar', 1.0)

            # Fetch data from Semantic Scholar API
            url = f"https://api.semanticscholar.org/v1/paper/{doi}"
            headers = {
                'User-Agent': f'ArticleMetadataRetriever/1.0 (mailto:{self.email})'
            }

            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()

                # Get title
                if 'title' in data:
                    metadata['title'] = data['title']

                # Get abstract
                if 'abstract' in data and data['abstract']:
                    metadata['abstract'] = data['abstract']

                # Get journal
                if 'venue' in data:
                    metadata['journal'] = data['venue']

                # Get authors
                if 'authors' in data and data['authors']:
                    author_list = [author['name'] for author in data['authors']]
                    metadata['authors'] = ', '.join(author_list)

                # Get publication date
                if 'year' in data:
                    metadata['publication_date'] = str(data['year'])

                metadata['success'] = bool(metadata['title'] or metadata['abstract'])

                # Cache the result
                if doi not in self.metadata_cache:
                    self.metadata_cache[doi] = {}

                self.metadata_cache[doi]['semanticscholar'] = metadata

        except Exception as e:
            self.logger.warning(f"Error retrieving Semantic Scholar metadata for DOI {doi}: {str(e)[:100]}...")
            self.errors['metadata_retrieval'].append({
                'doi': doi,
                'source': 'semanticscholar',
                'error': str(e)
            })

        return metadata

    def get_best_metadata(self, doi: str) -> Dict:
        """
        Get the best available metadata by trying multiple sources.

        Args:
            doi (str): DOI of the article

        Returns:
            Dict: Combined metadata from best sources
        """
        combined_metadata = {
            'title': None,
            'abstract': None,
            'journal': None,
            'authors': None,
            'publication_date': None,
            'pmid': None,
            'doi': doi,
            'metadata_sources': []
        }

        # Try PubMed first (usually has good abstracts)
        pubmed_data = self.get_pubmed_metadata(doi)
        if pubmed_data['success']:
            combined_metadata.update({
                'title': pubmed_data['title'],
                'abstract': pubmed_data['abstract'],
                'journal': pubmed_data['journal'],
                'authors': pubmed_data['authors'],
                'publication_date': pubmed_data['publication_date'],
                'pmid': pubmed_data['pmid']
            })
            combined_metadata['metadata_sources'].append('pubmed')

        # Try Crossref next
        crossref_data = self.get_crossref_metadata(doi)
        if crossref_data['success']:
            # Fill in missing fields
            if not combined_metadata['title'] and crossref_data['title']:
                combined_metadata['title'] = crossref_data['title']
            if not combined_metadata['abstract'] and crossref_data['abstract']:
                combined_metadata['abstract'] = crossref_data['abstract']
            if not combined_metadata['journal'] and crossref_data['journal']:
                combined_metadata['journal'] = crossref_data['journal']
            if not combined_metadata['authors'] and crossref_data['authors']:
                combined_metadata['authors'] = crossref_data['authors']
            if not combined_metadata['publication_date'] and crossref_data['publication_date']:
                combined_metadata['publication_date'] = crossref_data['publication_date']

            if 'crossref' not in combined_metadata['metadata_sources']:
                combined_metadata['metadata_sources'].append('crossref')

        # Try Semantic Scholar last
        semanticscholar_data = self.get_semanticscholar_metadata(doi)
        if semanticscholar_data['success']:
            # Fill in missing fields
            if not combined_metadata['title'] and semanticscholar_data['title']:
                combined_metadata['title'] = semanticscholar_data['title']
            if not combined_metadata['abstract'] and semanticscholar_data['abstract']:
                combined_metadata['abstract'] = semanticscholar_data['abstract']
            if not combined_metadata['journal'] and semanticscholar_data['journal']:
                combined_metadata['journal'] = semanticscholar_data['journal']
            if not combined_metadata['authors'] and semanticscholar_data['authors']:
                combined_metadata['authors'] = semanticscholar_data['authors']
            if not combined_metadata['publication_date'] and semanticscholar_data['publication_date']:
                combined_metadata['publication_date'] = semanticscholar_data['publication_date']

            if 'semanticscholar' not in combined_metadata['metadata_sources']:
                combined_metadata['metadata_sources'].append('semanticscholar')

        # Track success/failure
        if combined_metadata['title'] or combined_metadata['abstract']:
            self.successful_retrievals += 1
        else:
            self.failed_retrievals += 1

        return combined_metadata

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a DataFrame to find DOIs and retrieve metadata, with enhanced validation.

        Args:
            df (pd.DataFrame): DataFrame with article information

        Returns:
            pd.DataFrame: Updated DataFrame with DOIs and metadata
        """
        original_df = df.copy()

        # Check for duplicate rows that might cause issues
        if df.duplicated().any():
            self.logger.warning(f"Found {df.duplicated().sum()} duplicate rows in input data. Removing duplicates.")
            df = df.drop_duplicates()

        # Check for existing metadata to preserve
        has_existing_title = 'title' in df.columns and df['title'].notna().any()
        has_existing_abstract = 'abstract' in df.columns and df['abstract'].notna().any()

        # Start with DOI identification
        self.logger.info("Step 1: Identifying DOIs for articles...")
        df_with_dois = self.find_dois(df)

        # Now retrieve metadata for articles with valid DOIs
        self.logger.info("Step 2: Retrieving metadata for articles with valid DOIs...")

        # Create a copy to avoid SettingWithCopyWarning
        result_df = df_with_dois.copy()

        # Initialize metadata columns if they don't exist
        for col in ['title', 'abstract', 'journal', 'authors', 'publication_date', 'pmid', 'metadata_sources']:
            if col not in result_df.columns:
                result_df[col] = None

        # Create a Record column if it doesn't exist
        if 'Record' not in result_df.columns:
            result_df['Record'] = None

        # Process records with valid DOIs, excluding any that might have caused the problem
        valid_doi_mask = result_df['doi'].apply(lambda x: self.is_valid_doi(x) if pd.notna(x) else False)

        # Count DOI frequencies
        doi_counts = result_df.loc[valid_doi_mask, 'doi'].value_counts()
        suspicious_dois = doi_counts[doi_counts > 3].index.tolist()

        if suspicious_dois:
            self.logger.warning(f"Found {len(suspicious_dois)} suspicious DOIs assigned to multiple (>3) records.")
            self.logger.warning(
                "These may indicate incorrect DOI assignments and will be skipped for metadata retrieval.")
            for doi in suspicious_dois:
                self.logger.warning(f"Suspicious DOI: {doi} - assigned to {doi_counts[doi]} records")

            # Exclude suspicious DOIs
            valid_doi_mask = valid_doi_mask & ~result_df['doi'].isin(suspicious_dois)

        records_to_process = result_df[valid_doi_mask].copy()

        if len(records_to_process) > 0:
            self.logger.info(f"Retrieving metadata for {len(records_to_process)} articles with valid DOIs")

            # Track which DOIs we've processed to avoid duplicates
            processed_dois = set()

            # Progress bar
            with tqdm(total=len(records_to_process), desc="Retrieving metadata") as pbar:
                for idx, row in records_to_process.iterrows():
                    doi = row['doi']

                    # Skip if we've already processed this DOI to avoid duplicates
                    if doi in processed_dois:
                        pbar.update(1)
                        continue

                    processed_dois.add(doi)

                    # Get best available metadata
                    metadata = self.get_best_metadata(doi)

                    # Find all rows with this DOI
                    doi_rows = result_df[result_df['doi'] == doi].index

                    for target_idx in doi_rows:
                        # Preserve existing metadata if available
                        if has_existing_title and pd.notna(result_df.at[target_idx, 'title']):
                            metadata['title'] = result_df.at[target_idx, 'title']

                        if has_existing_abstract and pd.notna(result_df.at[target_idx, 'abstract']):
                            metadata['abstract'] = result_df.at[target_idx, 'abstract']

                        # Update DataFrame
                        for field in ['title', 'abstract', 'journal', 'authors', 'publication_date', 'pmid']:
                            if metadata[field]:
                                result_df.at[target_idx, field] = metadata[field]

                        # Update metadata sources
                        result_df.at[target_idx, 'metadata_sources'] = ', '.join(metadata['metadata_sources'])

                        # Update Record field (combining title and abstract)
                        title = metadata['title'] or ''
                        abstract = metadata['abstract'] or ''
                        record = f"{title} {abstract}".strip()
                        if record:
                            result_df.at[target_idx, 'Record'] = record

                    pbar.update(1)

        # If we have original title/abstract columns, use them to create Record where missing
        if 'Record' in result_df.columns:
            missing_record_mask = result_df['Record'].isna()
            for idx in result_df[missing_record_mask].index:
                title = ''
                abstract = ''

                if 'title' in result_df.columns and pd.notna(result_df.at[idx, 'title']):
                    title = result_df.at[idx, 'title']

                if 'abstract' in result_df.columns and pd.notna(result_df.at[idx, 'abstract']):
                    abstract = result_df.at[idx, 'abstract']

                if title or abstract:
                    result_df.at[idx, 'Record'] = f"{title} {abstract}".strip()

        # Report results
        valid_records = result_df['Record'].notna().sum()
        self.logger.info(f"Metadata retrieval complete.")
        self.logger.info(f"Articles with valid DOIs: {len(records_to_process)}")
        self.logger.info(f"Articles with title or abstract: {valid_records}")
        if len(result_df) > 0:
            self.logger.info(f"Success rate: {(valid_records / len(result_df) * 100):.1f}%")

        # Add a validation column to identify possible issues
        result_df['metadata_validation'] = 'ok'

        # Check for suspicious patterns (same metadata across many unrelated entries)
        if len(result_df) > 10:  # Only check for larger datasets
            for field in ['title', 'abstract']:
                if field in result_df.columns:
                    # Count frequencies
                    field_counts = result_df[field].value_counts()
                    # Flag if any metadata appears across >20% of entries
                    suspicious_threshold = max(3, len(result_df) * 0.2)
                    suspicious_values = field_counts[field_counts > suspicious_threshold].index.tolist()

                    if suspicious_values:
                        self.logger.warning(f"Found suspiciously common {field} values that appear in many records")
                        for value in suspicious_values:
                            affected_count = field_counts[value]
                            if pd.notna(value) and len(str(value)) > 20:  # Ensure it's a real value
                                self.logger.warning(
                                    f"Suspicious {field}: '{str(value)[:50]}...' appears {affected_count} times")
                                # Mark affected rows
                                affected_rows = result_df[result_df[field] == value].index
                                result_df.loc[affected_rows, 'metadata_validation'] = f'suspicious_{field}'

        return result_df

    def process_dataframe_with_checkpoint(self, df, checkpoint_path=None):
        """
        Process a DataFrame with checkpointing to avoid reprocessing items.

        Args:
            df (pd.DataFrame): DataFrame with article information
            checkpoint_path (str, optional): Path for checkpoint file

        Returns:
            pd.DataFrame: Updated DataFrame with DOIs and metadata
            dict: Summary statistics
        """
        import os

        # Create a unique identifier for each row if none exists
        if 'uniqueid' not in df.columns:
            self.logger.info("Creating uniqueid column for tracking")
            # Use title + authors + year if available, otherwise a hash of all columns
            if all(col in df.columns for col in ['Title', 'Authors', 'Year']):
                df['uniqueid'] = df.apply(
                    lambda row: hashlib.md5(f"{row['Title']}|{row['Authors']}|{str(row['Year'])}".encode()).hexdigest(),
                    axis=1
                )
            else:
                # Use all columns to create a uniqueid
                df['uniqueid'] = df.apply(
                    lambda row: hashlib.md5(str(row.values).encode()).hexdigest(),
                    axis=1
                )

        # Load checkpoint if it exists
        checkpoint_df = None
        if checkpoint_path:
            checkpoint_df = self.load_checkpoint(checkpoint_path)

        # If we have a checkpoint, determine which rows need processing
        if checkpoint_df is not None and 'uniqueid' in checkpoint_df.columns:
            # Find rows that haven't been processed yet
            processed_ids = set(checkpoint_df['uniqueid'])
            to_process = df[~df['uniqueid'].isin(processed_ids)].copy()

            self.logger.info(f"Found {len(processed_ids)} already processed articles in checkpoint")
            self.logger.info(f"Need to process {len(to_process)} new articles")

            if len(to_process) == 0:
                self.logger.info("All articles already processed, returning checkpoint data")
                summary = self.summarize_results(checkpoint_df)
                self.print_summary(summary)
                return checkpoint_df, summary

            # Process new rows
            new_results = self.process_dataframe(to_process)

            # Combine with checkpoint data
            result_df = pd.concat([checkpoint_df, new_results], ignore_index=True)

            # Remove any duplicates (by uniqueid)
            result_df = result_df.drop_duplicates(subset='uniqueid', keep='first')

        else:
            # Process the entire DataFrame
            result_df = self.process_dataframe(df)

        # Save checkpoint
        if checkpoint_path:
            self.save_checkpoint(result_df, checkpoint_path)

        # Generate summary
        summary = self.summarize_results(result_df)
        self.print_summary(summary)

        return result_df, summary

    def summarize_results(self, df):
        """
        Generate a detailed summary of processing results.

        Args:
            df (pd.DataFrame): The processed DataFrame

        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_articles': len(df),
            'doi_success': 0,
            'doi_failure': 0,
            'metadata_success': 0,
            'metadata_failure': 0,
            'suspicious_entries': 0
        }

        # Count DOI successes/failures
        has_valid_doi = df['doi'].apply(lambda x: self.is_valid_doi(x) if pd.notna(x) else False)
        summary['doi_success'] = has_valid_doi.sum()
        summary['doi_failure'] = len(df) - summary['doi_success']

        # Count metadata successes/failures
        has_title = df['title'].notna()
        has_abstract = df['abstract'].notna()
        has_metadata = has_title | has_abstract
        summary['metadata_success'] = has_metadata.sum()
        summary['metadata_failure'] = len(df) - summary['metadata_success']

        # Count suspicious entries
        if 'metadata_validation' in df.columns:
            summary['suspicious_entries'] = (df['metadata_validation'] != 'ok').sum()

        # Additional details
        if 'doi_source' in df.columns:
            doi_sources = df['doi_source'].value_counts().to_dict()
            summary['doi_sources'] = doi_sources

        if 'metadata_sources' in df.columns:
            # Split the comma-separated values
            all_sources = []
            for sources in df['metadata_sources'].dropna():
                if isinstance(sources, str):
                    all_sources.extend([s.strip() for s in sources.split(',')])

            from collections import Counter
            summary['metadata_sources'] = dict(Counter(all_sources))

        return summary

    def print_summary(self, summary):
        """Print a formatted summary to the console."""
        print("\n===== ARTICLE METADATA RETRIEVAL SUMMARY =====")
        print(f"Total articles processed: {summary['total_articles']}")
        print(f"DOI identification:")
        print(
            f"  ✓ Success: {summary['doi_success']} articles ({summary['doi_success'] / summary['total_articles'] * 100:.1f}%)")
        print(
            f"  ✗ Failure: {summary['doi_failure']} articles ({summary['doi_failure'] / summary['total_articles'] * 100:.1f}%)")
        print(f"Metadata retrieval:")
        print(
            f"  ✓ Success: {summary['metadata_success']} articles ({summary['metadata_success'] / summary['total_articles'] * 100:.1f}%)")
        print(
            f"  ✗ Failure: {summary['metadata_failure']} articles ({summary['metadata_failure'] / summary['total_articles'] * 100:.1f}%)")

        if 'suspicious_entries' in summary:
            print(f"Suspicious entries: {summary['suspicious_entries']}")

        if 'doi_sources' in summary:
            print("\nDOI sources:")
            for source, count in sorted(summary['doi_sources'].items(), key=lambda x: x[1], reverse=True):
                if pd.notna(source):
                    print(f"  {source}: {count}")

        if 'metadata_sources' in summary:
            print("\nMetadata sources:")
            for source, count in sorted(summary['metadata_sources'].items(), key=lambda x: x[1], reverse=True):
                if pd.notna(source):
                    print(f"  {source}: {count}")

        print("=================================================")

    def load_checkpoint(self, checkpoint_path):
        """
        Load a previously saved checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file

        Returns:
            pd.DataFrame: Checkpoint data or None if not found
        """
        if os.path.exists(checkpoint_path):
            try:
                checkpoint_df = pd.read_excel(checkpoint_path)
                self.logger.info(
                    f"Loaded checkpoint with {len(checkpoint_df)} processed articles from {checkpoint_path}")
                return checkpoint_df
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")

        self.logger.info(f"No checkpoint found at {checkpoint_path}")
        return None

    def save_checkpoint(self, df, checkpoint_path):
        """
        Save the current state as a checkpoint.

        Args:
            df (pd.DataFrame): The DataFrame to save
            checkpoint_path (str): Path to save the checkpoint

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            df.to_excel(checkpoint_path, index=False)
            self.logger.info(f"Saved checkpoint with {len(df)} processed articles to {checkpoint_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")
            return False

# Usage example
if __name__ == "__main__":
    # Initialize the retriever
    retriever = ArticleMetadataRetriever(
        email="your_email@example.com",
        verbose=True
    )

    # Load your dataset
    input_file = "your_input_file.xlsx"
    df = pd.read_excel(input_file)

    # Process the dataset
    result_df = retriever.process_dataframe(df)

    # Save the results
    output_file = "processed_articles.xlsx"
    result_df.to_excel(output_file, index=False)
    print(f"Processed data saved to {output_file}")

    # Summary of errors
    retriever.summarize_errors()