"""
PDF download utility for Academate.
"""

import os
import pickle
import requests
import pandas as pd
import logging
from tqdm import tqdm
import time
import random


class PDFDownloader:
    """PDF downloader for systematic reviews."""

    def __init__(self, output_directory, email="user@example.com", logger=None, verbose=False):
        """
        Initialize the PDF downloader.

        Args:
            output_directory (str): Directory to save PDFs
            email (str, optional): Email for API access. Defaults to "user@example.com".
            logger (logging.Logger, optional): Logger instance. Defaults to None.
            verbose (bool, optional): Verbose output. Defaults to False.
        """
        self.output_directory = output_directory
        self.email = email
        self.verbose = verbose

        # Set up logger
        if logger:
            self.logger = logger
        else:
            self.logger = self._setup_logger()

        # Create output directory
        os.makedirs(output_directory, exist_ok=True)

        # Initialize error tracking
        self.pdf_download_error = set()

    def _setup_logger(self):
        """Set up logger for this downloader."""
        logger = logging.getLogger("PDFDownloader")
        level = logging.DEBUG if self.verbose else logging.INFO
        logger.setLevel(level)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False

        return logger

    def download_pdf(self, doi, output_path):
        """
        Download a PDF for a given DOI.

        Args:
            doi (str): DOI of the article
            output_path (str): Path to save the PDF

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try Unpaywall API first
            success = self._download_from_unpaywall(doi, output_path)
            if success:
                return True

            # Try DOI.org next
            success = self._download_from_doi_org(doi, output_path)
            if success:
                return True

            # Try direct publishers as a last resort
            success = self._download_from_publishers(doi, output_path)
            if success:
                return True

            # If all methods fail
            self.logger.warning(f"Could not download PDF for DOI: {doi}")
            return False

        except Exception as e:
            self.logger.error(f"Error downloading PDF for DOI {doi}: {str(e)}")
            return False

    def _download_from_unpaywall(self, doi, output_path):
        """
        Download PDF using Unpaywall API.

        Args:
            doi (str): DOI of the article
            output_path (str): Path to save the PDF

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Unpaywall API endpoint
            url = f"https://api.unpaywall.org/v2/{doi}?email={self.email}"

            # Get metadata
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Check for open access PDF link
                if data.get('is_oa') and 'best_oa_location' in data and data['best_oa_location']:
                    pdf_url = data['best_oa_location'].get('url_for_pdf')

                    if pdf_url:
                        # Download the PDF
                        pdf_response = requests.get(pdf_url, timeout=30, stream=True)

                        if pdf_response.status_code == 200:
                            with open(output_path, 'wb') as f:
                                for chunk in pdf_response.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)

                            self.logger.debug(f"Downloaded PDF for {doi} from Unpaywall")
                            return True

            return False

        except Exception as e:
            self.logger.debug(f"Unpaywall download failed for {doi}: {str(e)}")
            return False

    def _download_from_doi_org(self, doi, output_path):
        """
        Download PDF using DOI.org resolution.

        Args:
            doi (str): DOI of the article
            output_path (str): Path to save the PDF

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # DOI resolver
            url = f"https://doi.org/{doi}"

            # Set up headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }

            # Resolve DOI to actual page
            response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)

            if response.status_code == 200:
                # Extract potential PDF links (this is a simplistic approach)
                html = response.text.lower()
                if 'pdf' in html or '.pdf' in response.url:
                    # Try to download PDF from the resolved URL
                    pdf_url = response.url
                    if not pdf_url.endswith('.pdf'):
                        # Try to find PDF link in page
                        import re
                        pdf_links = re.findall(r'href="([^"]*\.pdf)"', html)
                        if pdf_links:
                            # Take the first PDF link
                            pdf_url = pdf_links[0]
                            if not pdf_url.startswith('http'):
                                # Handle relative URLs
                                base_url = response.url.rsplit('/', 1)[0]
                                pdf_url = f"{base_url}/{pdf_url}"

                    # Download the PDF
                    pdf_response = requests.get(pdf_url, headers=headers, timeout=30, stream=True)

                    if pdf_response.status_code == 200 and pdf_response.headers.get('Content-Type', '').startswith(
                            'application/pdf'):
                        with open(output_path, 'wb') as f:
                            for chunk in pdf_response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)

                        self.logger.debug(f"Downloaded PDF for {doi} from DOI.org resolution")
                        return True

            return False

        except Exception as e:
            self.logger.debug(f"DOI.org download failed for {doi}: {str(e)}")
            return False

    def _download_from_publishers(self, doi, output_path):
        """
        Download PDF by trying known publisher patterns.

        Args:
            doi (str): DOI of the article
            output_path (str): Path to save the PDF

        Returns:
            bool: True if successful, False otherwise
        """
        # This would be a more complex implementation that tries known URL patterns
        # for different publishers based on the DOI prefix
        return False

    def process_dataframe(self, df, checkpoint_file=None):
        """
        Process a DataFrame to download PDFs for all articles.

        Args:
            df (pd.DataFrame): DataFrame with articles
            checkpoint_file (str, optional): Path to save progress. Defaults to None.

        Returns:
            pd.DataFrame: Updated DataFrame with PDF paths
        """
        # Make a copy of the DataFrame
        df = df.copy()

        # Add pdf_path and pdf_name columns if they don't exist
        if 'pdf_path' not in df.columns:
            df['pdf_path'] = None
        if 'pdf_name' not in df.columns:
            df['pdf_name'] = None

        # Load checkpoint if exists
        download_progress = {}
        if checkpoint_file and os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                download_progress = pickle.load(f)

            # Apply progress to DataFrame
            for uniqueid, info in download_progress.items():
                mask = df['uniqueid'] == uniqueid
                if mask.any():
                    df.loc[mask, 'pdf_path'] = info.get('pdf_path')
                    df.loc[mask, 'pdf_name'] = info.get('pdf_name')

        # Count how many are already downloaded
        already_downloaded = df['pdf_path'].notna().sum()
        self.logger.info(f"Already downloaded: {already_downloaded}/{len(df)} PDFs")

        # Get records that need download
        records_to_download = df[df['pdf_path'].isna()].to_dict('records')

        if not records_to_download:
            self.logger.info("No new PDFs to download")
            return df

        self.logger.info(f"Downloading {len(records_to_download)} PDFs")

        # Process each record
        for record in tqdm(records_to_download, desc="Downloading PDFs"):
            uniqueid = record['uniqueid']

            # Skip if already processed in checkpoint
            if uniqueid in download_progress:
                continue

            # Get DOI
            doi = record.get('doi')
            if not doi:
                self.logger.warning(f"No DOI found for record {uniqueid}")
                self.pdf_download_error.add(uniqueid)
                download_progress[uniqueid] = {'pdf_path': None, 'pdf_name': None}
                continue

            # Create PDF filename
            pdf_name = doi.replace('/', '_').replace('\\', '_').replace(':', '_') + '.pdf'
            pdf_path = os.path.join(self.output_directory, pdf_name)

            # Download PDF
            success = self.download_pdf(doi, pdf_path)

            if success:
                # Update DataFrame
                mask = df['uniqueid'] == uniqueid
                df.loc[mask, 'pdf_path'] = pdf_path
                df.loc[mask, 'pdf_name'] = pdf_name

                # Update progress
                download_progress[uniqueid] = {'pdf_path': pdf_path, 'pdf_name': pdf_name}
            else:
                self.logger.warning(f"Failed to download PDF for record {uniqueid}")
                self.pdf_download_error.add(uniqueid)
                download_progress[uniqueid] = {'pdf_path': None, 'pdf_name': None}

            # Save checkpoint periodically
            if checkpoint_file and len(download_progress) % 10 == 0:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(download_progress, f)

            # Small delay to avoid overwhelming servers
            time.sleep(random.uniform(1, 3))

        # Final checkpoint save
        if checkpoint_file:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(download_progress, f)

        self.logger.info(f"Downloaded {df['pdf_path'].notna().sum() - already_downloaded} new PDFs")
        self.logger.info(f"Total PDFs available: {df['pdf_path'].notna().sum()}/{len(df)}")
        self.logger.info(f"Total download errors: {len(self.pdf_download_error)}")

        return df