import os
import sys
import re
import logging
import requests
import subprocess
import shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from habanero import Crossref
from paperscraper.pdf import save_pdf
from doi2pdf import doi2pdf
from typing import Optional, Dict, List, Union
import PyPDF2
from Bio import Entrez  # For interacting with PubMed

# Configure logging for httpx, httpcore, requests, and urllib3
httpx_logger = logging.getLogger("httpx")
httpcore_logger = logging.getLogger("httpcore")
requests_logger = logging.getLogger("requests")
urllib3_logger = logging.getLogger("urllib3")

httpx_logger.setLevel(logging.WARNING)
httpcore_logger.setLevel(logging.WARNING)
requests_logger.setLevel(logging.WARNING)
urllib3_logger.setLevel(logging.WARNING)

class ArticleDownloader:
    """
    A class to handle the downloading of academic articles from various sources.
    Designed to work with the academate class.
    """

    def __init__(self, output_directory: str, email: str, logger=None, verbose: bool = False):
        """
        Initialize the ArticleDownloader.

        Args:
            output_directory (str): Directory where PDFs will be saved
            email (str): Email for API authentication
            logger: Logger instance from parent class
            verbose (bool): Whether to show detailed logging
        """
        self.output_directory = Path(output_directory)
        self.email = email
        self.verbose = verbose
        self.logger = logger or logging.getLogger(__name__)

        # Create output directory if it doesn't exist
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Initialize error tracking
        self.download_errors = {}
        self.pdf_download_error = set()  # Compatible with academate's screening2_PDFdownload_error

    @staticmethod
    def is_valid_doi(doi: str) -> bool:
        """Check if a DOI string matches the expected format."""
        doi_regex = re.compile(r'^10.\d{4,9}/[-._;()/:A-Z0-9]+$', re.IGNORECASE)
        return bool(doi_regex.match(doi))

    @staticmethod
    def is_tool(name: str) -> bool:
        """Check if a command-line tool is available."""
        from shutil import which
        return which(name) is not None

    def deduplicate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # We sort so that "best" rows (downloaded, valid pdf_path, etc.) come first
        df_sorted = df.sort_values(
            by=['uniqueid', 'downloaded', 'pdf_path', 'pmid', 'doi'],
            ascending=[True, False, False, False, False]
        )
        df_deduped = df_sorted.drop_duplicates(subset='uniqueid', keep='first')
        return df_deduped

    def count_valid_pdfs(self, df: pd.DataFrame) -> int:
        """Counts the number of articles with valid downloaded PDFs."""
        count = 0
        for index, row in df.iterrows():
            pdf_path = row['pdf_path']
            # Only count as valid if pdf_path is not None or nan
            if not pd.isna(pdf_path) and os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                try:
                    with open(pdf_path, 'rb') as f:
                        PyPDF2.PdfReader(f)  # Check if it's a valid PDF
                    count += 1
                except Exception as e:
                    self.logger.error(f"Invalid PDF encountered at {pdf_path}: {e}")
                    pass  # Not a valid PDF, don't count
        return count

    def cleanup_output_directory(self, df: pd.DataFrame):
        """
        Performs cleanup operations on the output directory after PDF downloads.

        Ensures that:
            1. Each row in the DataFrame corresponds to at most one PDF file.
            2. Rows without a valid PDF have pdf_path set to None.
            3. Extra PDFs not matching any pdf_name in the DataFrame are removed.
        """
        # Get list of expected PDF filenames from the DataFrame
        expected_pdf_names = set(df['pdf_name'].dropna())

        # Check for extra PDFs and delete them
        for filename in os.listdir(self.output_directory):
            filepath = os.path.join(self.output_directory, filename)
            if os.path.isfile(filepath) and filepath.endswith(".pdf"):
                if filename not in expected_pdf_names:
                    try:
                        os.remove(filepath)
                        self.logger.info(f"Deleted extra PDF: {filename}")
                    except Exception as e:
                        self.logger.error(f"Failed to delete {filename}: {e}")

        # Update pdf_path to None for rows without a valid PDF
        for index, row in df.iterrows():
            if pd.notna(row['pdf_name']):
                pdf_path = os.path.join(self.output_directory, row['pdf_name'])
                df.at[index, 'pdf_path'] = pdf_path  # Update to actual path

                # Validate PDF
                if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
                    self.logger.error(f"PDF not found or empty: {pdf_path}")
                    df.at[index, 'pdf_path'] = None  # Set to None if invalid
                    df.at[index, 'error_message'] = "PDF not found or empty"
                else:
                    try:
                        with open(pdf_path, 'rb') as f:
                            PyPDF2.PdfReader(f)  # Check for valid PDF structure
                        self.logger.info(f"Valid PDF: {pdf_path}")
                    except Exception as e:
                        self.logger.error(f"Invalid or inaccessible PDF: {pdf_path} - {e}")
                        df.at[index, 'pdf_path'] = None  # Set to None if invalid
                        df.at[index, 'error_message'] = f"Invalid PDF: {e}"
            else:
                df.at[index, 'pdf_path'] = None  # Ensure None if pdf_name is missing

    def find_dois(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Finds DOIs for articles in a DataFrame using multiple strategies.

        1. Checks existing columns for DOIs (excluding 'doi' itself).
        2. Uses Crossref API to find missing DOIs, with retries.
        3. Avoids re-estimating DOIs that are already found.
        4. Handles multiple runs by not processing rows with existing DOIs.
        5. **NEW:** Logs found DOIs and their sources for better tracking.
        """
        working_df = df.copy()

        if 'doi' not in working_df.columns:
            working_df['doi'] = None
        if 'open_access' not in working_df.columns:
            working_df['open_access'] = None
        if 'doi_source' not in working_df.columns:
            working_df['doi_source'] = None  # Track where the DOI was found

        # 1. Check existing columns for DOIs
        for col in working_df.columns:
            if col != 'doi':
                for idx, row in working_df.iterrows():
                    if pd.isna(row['doi']) or row['doi'] == 'DOI not found':
                        potential_doi = str(row[col])
                        if self.is_valid_doi(potential_doi):
                            working_df.at[idx, 'doi'] = potential_doi
                            working_df.at[idx, 'doi_source'] = col  # Log the source column
                            if self.verbose:
                                self.logger.info(f"Found DOI in column '{col}': {potential_doi}")

        # 2. Use Crossref for remaining missing DOIs
        records_to_process = working_df[working_df['doi'].isna() | (working_df['doi'] == 'DOI not found')]
        if len(records_to_process) > 0:
            cr = Crossref()

            def get_doi(row):
                query_parts = []
                for col in working_df.columns:
                    if col not in ['doi', 'open_access', 'uniqueid', 'pdf_name', 'pdf_path',
                                   'doi_source'] and pd.notna(row[col]):
                        query_parts.append(str(row[col]))
                query = ' '.join(query_parts)

                for attempt in range(3):
                    try:
                        result = cr.works(query=query)
                        items = result['message']['items']
                        if items:
                            doi = items[0].get('DOI', 'DOI not found')
                            if self.is_valid_doi(doi):
                                return doi, 'crossref'  # Return DOI and source
                            else:
                                return 'DOI not found', None
                        else:
                            return 'DOI not found', None
                    except Exception as e:
                        self.logger.error(f"DOI search attempt {attempt + 1} failed for query '{query}': {e}")
                return 'DOI not found', None

            tqdm.pandas(desc="Finding DOIs with Crossref")
            # Apply the function and unpack the returned tuple into 'doi' and 'doi_source'
            records_to_process[['doi', 'doi_source']] = records_to_process.progress_apply(
                get_doi, axis=1, result_type='expand'
            )
            working_df.update(records_to_process[['doi', 'doi_source']])

        return working_df

    def find_pmids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Finds PMIDs for articles in a DataFrame using the following strategies:
        1. Checks existing columns for PMIDs.
        2. Uses the PubMed API (via Entrez from Biopython) to find missing PMIDs based on the title.
        """
        working_df = df.copy()

        if 'pmid' not in working_df.columns:
            working_df['pmid'] = None
        if 'pmid_source' not in working_df.columns:
            working_df['pmid_source'] = None

        # 1. Check existing columns for PMIDs
        for col in working_df.columns:
            if col != 'pmid':
                for idx, row in working_df.iterrows():
                    if pd.isna(row['pmid']):
                        potential_pmid = str(row[col])
                        if potential_pmid.isdigit():  # Simple check: PMIDs are typically integers
                            working_df.at[idx, 'pmid'] = potential_pmid
                            working_df.at[idx, 'pmid_source'] = col
                            if self.verbose:
                                self.logger.info(f"Found PMID in column '{col}': {potential_pmid}")

        # 2. Use PubMed API (Entrez) for remaining missing PMIDs
        records_to_process = working_df[working_df['pmid'].isna()]
        if len(records_to_process) > 0:
            Entrez.email = self.email  # Provide your email to Entrez

            def get_pmid(row):
                title = row['title']
                if pd.isna(title):
                    return 'PMID not found', None

                for attempt in range(3):  # Retry up to 3 times
                    try:
                        # Use esearch to find PMIDs based on the title
                        handle = Entrez.esearch(db="pubmed", term=title, retmax=1)
                        record = Entrez.read(handle)
                        handle.close()

                        if record["IdList"]:
                            pmid = record["IdList"][0]
                            return pmid, 'pubmed'
                        else:
                            return 'PMID not found', None
                    except Exception as e:
                        self.logger.error(f"PMID search attempt {attempt + 1} failed for title '{title}': {e}")
                return 'PMID not found', None

            tqdm.pandas(desc="Finding PMIDs with PubMed")
            records_to_process[['pmid', 'pmid_source']] = records_to_process.progress_apply(
                get_pmid, axis=1, result_type='expand'
            )
            working_df.update(records_to_process[['pmid', 'pmid_source']])

        return working_df

    def download_article(self, doi: str, retries: int = 3) -> Optional[str]:
        """
        Downloads an article PDF using various methods, with retries.

        Args:
            doi (str): DOI of the article.
            retries (int): Number of times to retry each download method.

        Returns:
            Optional[str]: Path to the downloaded PDF, or None if all methods fail.
        """
        if not self.is_valid_doi(doi):
            self.logger.error(f"Invalid DOI format: {doi}")
            return None

        expected_filename = f"{doi.replace('/', '_')}.pdf"
        expected_pdf_path = self.output_directory / expected_filename

        # Check if already downloaded and has content
        if expected_pdf_path.exists() and expected_pdf_path.stat().st_size > 0:
            self.logger.info(f"PDF already exists for {doi}")
            return str(expected_pdf_path)

        download_methods = [
            ('unpaywall', self._download_unpaywall),
            ('paperscraper', self._download_paperscraper),
            ('doi2pdf', self._download_doi2pdf),
            ('scihub', self._download_scihub)
        ]

        for method_name, method in download_methods:
            # print("method_name", method_name)
            for attempt in range(retries):
                try:
                    pdf_path = method(doi, str(expected_pdf_path))
                    # Check if download was successful before returning
                    if pdf_path and os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                        self.logger.info(f"Successfully downloaded {doi} using {method_name} (attempt {attempt + 1})")
                        return pdf_path
                    else:
                        self.logger.error(
                            f"{method_name} returned an empty or invalid file for {doi} (attempt {attempt + 1})")

                except Exception as e:
                    self.logger.error(f"{method_name} failed for {doi} (attempt {attempt + 1}): {e}")
                    if doi not in self.download_errors:
                        self.download_errors[doi] = []
                    self.download_errors[doi].append(f"{method_name} (attempt {attempt + 1}): {str(e)}")
        return None

    def _download_unpaywall(self, doi: str, pdf_path: str) -> Optional[str]:
        """Download using Unpaywall."""
        try:
            response = requests.get(
                f"https://api.unpaywall.org/v2/{doi}?email={self.email}"
            )
            data = response.json()
            for location in data.get('oa_locations', []):
                pdf_url = location.get('url_for_pdf')
                if pdf_url:
                    pdf_response = requests.get(pdf_url)
                    if pdf_response.status_code == 200:
                        with open(pdf_path, 'wb') as f:
                            f.write(pdf_response.content)
                        return pdf_path
        except Exception as e:
            raise Exception(f"Unpaywall download failed: {e}")
        return None

    def _download_paperscraper(self, doi: str, pdf_path: str) -> Optional[str]:
        """Download using Paperscraper."""
        try:
            save_pdf({'doi': doi}, pdf_path)
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                return pdf_path
        except Exception as e:
            raise Exception(f"Paperscraper download failed: {e}")
        return None

    def _download_doi2pdf(self, doi: str, pdf_path: str) -> Optional[str]:
        """Download using doi2pdf."""
        try:
            os.environ['SCI_HUB_URL'] = 'https://sci-hub.wf/'
            result = doi2pdf(doi, output=pdf_path)
            if os.path.exists(result) and os.path.getsize(result) > 0:
                return result
        except Exception as e:
            raise Exception(f"doi2pdf download failed: {e}")
        return None

    def _download_scihub(self, doi: str, pdf_path: str) -> Optional[str]:
        """Download using Sci-Hub mirrors."""
        mirrors = ['https://sci-hub.se', 'https://sci-hub.st', 'https://sci-hub.ru']
        for mirror in mirrors:
            try:
                response = requests.get(f"{mirror}/{doi}")
                if response.status_code == 200:
                    pdf_url = None
                    for line in response.text.split('\n'):
                        if 'iframe src' in line:
                            pdf_url = line.split('"')[1]
                            break
                    if pdf_url:
                        if not pdf_url.startswith('http'):
                            pdf_url = f"{mirror}{pdf_url}"
                        pdf_response = requests.get(pdf_url)
                        if pdf_response.status_code == 200:
                            with open(pdf_path, 'wb') as f:
                                f.write(pdf_response.content)
                            if os.path.getsize(pdf_path) > 0:
                                return pdf_path
            except Exception:
                continue
        raise Exception("All Sci-Hub mirrors failed")

    def _download_pubmed(self, pmid: str, pdf_path: str) -> Optional[str]:
        """
        Downloads an article PDF using its PubMed ID (PMID).

        Args:
            pmid (str): The PubMed ID of the article.
            pdf_path (str): The path where the PDF should be saved.

        Returns:
            Optional[str]: The path to the downloaded PDF, or None if the download failed.
        """
        try:
            # Fetch the Entrez record for the PMID
            handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml", email=self.email)
            record = Entrez.read(handle)
            handle.close()

            # Extract the DOI from the Entrez record, if available
            article = record['PubmedArticle'][0]['MedlineCitation']['Article']
            if 'ELocationID' in article:
                for e_id in article['ELocationID']:
                    if e_id.attributes['EIdType'] == 'doi':
                        doi = str(e_id)
                        self.logger.info(f"Found DOI {doi} for PMID {pmid}")

                        # Try downloading using existing methods with the found DOI
                        download_methods = [
                            ('unpaywall', self._download_unpaywall),
                            ('paperscraper', self._download_paperscraper),
                            ('doi2pdf', self._download_doi2pdf),
                            ('scihub', self._download_scihub)
                        ]

                        for method_name, method in download_methods:
                            try:
                                pdf_path = method(doi=doi, pdf_path=pdf_path)
                                if pdf_path and os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                                    self.logger.info(f"Successfully downloaded {doi} using {method_name}")
                                    return pdf_path
                            except Exception as e:
                                self.logger.error(f"{method_name} failed for {doi}: {e}")

                        self.logger.error(f"All download methods failed for DOI {doi} (PMID {pmid})")
                        return None

            self.logger.error(f"DOI not found in PubMed record for PMID {pmid}")
            return None

        except Exception as e:
            self.logger.error(f"Error downloading with PMID {pmid}: {e}")
            return None

    def process_dataframe(self, df: pd.DataFrame, checkpoint_file: Optional[str] = None) -> pd.DataFrame:
        """
        Processes a DataFrame to find DOIs and download PDFs, with persistent state across runs.
        Also, deduplicates the DataFrame based on 'record', keeping the most complete row.

        Args:
            df (pd.DataFrame): DataFrame with article information.
            checkpoint_file (Optional[str]): Path to save a checkpoint pickle file.

        Returns:
            pd.DataFrame: Updated DataFrame with DOI, PDF path, and download status.
        """
        df = df.copy()

        # Initialize required columns, including 'pmid' and 'pmid_source'
        for col in ['pdf_path', 'pdf_name', 'download_attempt', 'downloaded', 'error_message', 'doi_source', 'pmid', 'pmid_source']:
            if col not in df.columns:
                df[col] = pd.NA

        # Convert columns to appropriate types
        df['download_attempt'] = df['download_attempt'].fillna(0).astype('Int64')
        df['downloaded'] = df['downloaded'].fillna(False).astype(bool)

        # Load checkpoint if it exists
        if checkpoint_file and os.path.exists(checkpoint_file):
            checkpoint_df = pd.read_pickle(checkpoint_file)

            # Ensure checkpoint_df isn't itself duplicated
            checkpoint_df.drop_duplicates(subset='uniqueid', keep='last', inplace=True)

            if 'uniqueid' in checkpoint_df.columns:
                df = pd.merge(
                    df,
                    checkpoint_df[
                        ['uniqueid', 'doi', 'pmid', 'download_attempt', 'downloaded', 'pdf_path', 'pdf_name']],
                    on='uniqueid',
                    how='left',
                    suffixes=('', '_y')
                )

                for col in ['download_attempt', 'downloaded', 'pdf_path', 'pdf_name', 'doi', 'pmid']:
                    if col + '_y' in df.columns:
                        df[col] = df[col].combine_first(df[col + '_y'])
                        df.drop(col + '_y', axis=1, inplace=True)

            # Now drop duplicates in the merged df
            df.drop_duplicates(subset='uniqueid', keep='first', inplace=True)

        # Find DOIs if needed, but only for rows that need it
        if 'doi' not in df.columns or df['doi'].isna().any() or (df['doi'] == 'DOI not found').any():
            df = self.find_dois(df)

        # Find PMIDs if needed
        if 'pmid' not in df.columns or df['pmid'].isna().any():
            df = self.find_pmids(df)

        # Process each article
        total_articles = len(df)
        with tqdm(total=total_articles, desc="Processing articles") as pbar:
            for idx, row in df.iterrows():
                uniqueid = str(row['uniqueid'])

                # Skip if PDF exists (and has content) or if it has already been downloaded successfully
                if (not pd.isna(row['pdf_path']) and os.path.exists(row['pdf_path']) and os.path.getsize(
                        row['pdf_path']) > 0) or row['downloaded'] == True:
                    pbar.update()
                    continue

                # Skip if DOI is invalid
                if pd.isna(row['doi']) or row['doi'] == 'DOI not found':
                    df.at[idx, 'error_message'] = 'Invalid DOI'
                    self.pdf_download_error.add(uniqueid)
                    pbar.update()
                    continue

                # If download_attempt is NA (new row), initialize it to 0
                if pd.isna(row['download_attempt']):
                    df.at[idx, 'download_attempt'] = 0

                # Increment the download attempt counter
                df.at[idx, 'download_attempt'] += 1

                # Download the article PDF using DOI
                pdf_path = self.download_article(row['doi'])

                # If DOI download failed, try downloading using PMID
                if not pdf_path and pd.notna(row['pmid']):
                    self.logger.info(f"Attempting download with PMID: {row['pmid']}")
                    pdf_path = self._download_pubmed(row['pmid'], row['pdf_path'])

                # Update DataFrame based on download outcome
                if pdf_path:
                    df.at[idx, 'downloaded'] = True
                    df.at[idx, 'pdf_path'] = pdf_path
                    df.at[idx, 'pdf_name'] = os.path.basename(pdf_path)
                    self.pdf_download_error.discard(uniqueid)
                else:
                    df.at[idx, 'downloaded'] = False
                    df.at[idx, 'error_message'] = str(self.download_errors.get(row['doi'], []))
                    self.pdf_download_error.add(uniqueid)

                # Save checkpoint if specified
                if checkpoint_file and idx % 10 == 0:
                    df.to_pickle(checkpoint_file)

                # Update progress bar description
                pbar.set_postfix({"Valid PDFs": f"{self.count_valid_pdfs(df)}/{total_articles}"}, refresh=False)
                pbar.update()

        # Perform cleanup of the output directory after all download attempts
        self.cleanup_output_directory(df)

        # Display the counts
        valid_pdf_count = self.count_valid_pdfs(df)
        total_articles = len(df)
        print(f"Successfully downloaded and validated PDFs: {valid_pdf_count} out of {total_articles}")

        # Deduplicate the DataFrame based on 'record' after processing
        df = self.deduplicate_dataframe(df)

        return df

