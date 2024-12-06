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

    def find_dois(self, df: pd.DataFrame) -> pd.DataFrame:
        """DOI finding logic - same as your existing find_record_doi method"""
        working_df = df.copy()

        # Initialize DOI and open_access columns
        if 'doi' not in working_df.columns:
            working_df['doi'] = None
        if 'open_access' not in working_df.columns:
            working_df['open_access'] = None

        # Process records that need DOI retrieval
        records_to_process = working_df[working_df['doi'].isna()]

        if len(records_to_process) > 0:
            cr = Crossref()

            def get_doi(row):
                query_parts = []
                for col in working_df.columns:
                    if col not in ['doi', 'open_access', 'uniqueid'] and pd.notna(row[col]):
                        query_parts.append(str(row[col]))
                query = ' '.join(query_parts)

                for attempt in range(3):
                    try:
                        result = cr.works(query=query)
                        items = result['message']['items']
                        if items:
                            return items[0].get('DOI', 'DOI not found')
                    except Exception as e:
                        self.logger.error(f"DOI search attempt {attempt + 1} failed: {e}")
                return 'DOI not found'

            tqdm.pandas(desc="Finding DOIs")
            records_to_process['doi'] = records_to_process.progress_apply(get_doi, axis=1)
            working_df.update(records_to_process)

        return working_df

    def download_article(self, doi: str) -> Optional[str]:
        """
        Download an article using various methods.

        Args:
            doi (str): DOI of the article

        Returns:
            Optional[str]: Path to downloaded PDF or None if download failed
        """
        if not self.is_valid_doi(doi):
            self.logger.error(f"Invalid DOI format: {doi}")
            return None

        expected_filename = f"{doi.replace('/', '_')}.pdf"
        expected_pdf_path = self.output_directory / expected_filename

        # Check if already downloaded
        if expected_pdf_path.exists() and expected_pdf_path.stat().st_size > 0:
            self.logger.info(f"PDF already exists for {doi}")
            return str(expected_pdf_path)

        download_methods = [
            ('unpaywall', self._download_unpaywall),
            # ('paperscraper', self._download_paperscraper),
            # ('doi2pdf', self._download_doi2pdf),
            # ('scihub', self._download_scihub)
        ]

        for method_name, method in download_methods:
            try:
                pdf_path = method(doi, str(expected_pdf_path))
                if pdf_path:
                    self.logger.info(f"Successfully downloaded {doi} using {method_name}")
                    return pdf_path
            except Exception as e:
                self.logger.error(f"{method_name} failed for {doi}: {e}")
                if doi not in self.download_errors:
                    self.download_errors[doi] = []
                self.download_errors[doi].append(f"{method_name}: {str(e)}")

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
            save_pdf({'doi': doi}, pdf_path=pdf_path)
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

    def process_dataframe(self, df: pd.DataFrame, checkpoint_file: Optional[str] = None) -> pd.DataFrame:
        """
        Process a DataFrame of articles, finding DOIs and downloading PDFs.
        Modified to be compatible with academate's data structure.
        """
        df = df.copy()

        # Initialize required columns
        for col in ['pdf_path', 'pdf_name', 'download_attempt', 'downloaded', 'error_message']:
            if col not in df.columns:
                df[col] = pd.NA

        df['download_attempt'] = df['download_attempt'].fillna(0)
        df['downloaded'] = df['downloaded'].fillna(False)

        # Find DOIs if needed
        if 'doi' not in df.columns or df['doi'].isna().any():
            df = self.find_dois(df)

        # Process each article
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing articles"):
            uniqueid = str(row['uniqueid'])

            if pd.isna(row['doi']) or row['doi'] == 'DOI not found':
                df.at[idx, 'error_message'] = 'Invalid DOI'
                self.pdf_download_error.add(uniqueid)
                continue

            if not pd.isna(row['pdf_path']) and os.path.exists(row['pdf_path']):
                continue

            df.at[idx, 'download_attempt'] += 1
            pdf_path = self.download_article(row['doi'])

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

        return df

