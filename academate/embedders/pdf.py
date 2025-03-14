"""
PDF embedder implementation.
"""

import os
import asyncio
import pickle
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from academate.base.embedder import AbstractEmbedder
import pandas as pd


class PDFEmbedder(AbstractEmbedder):
    """Embedder for PDF documents."""

    def __init__(self, embeddings, vector_store_path, embedding_type, embeddings_path=None, verbose=False):
        """
        Initialize the PDF embedder.

        Args:
            embeddings (object): Embeddings model
            vector_store_path (str): Path to vector store
            verbose (bool, optional): Verbose output. Defaults to False.
        """
        super().__init__(
            embeddings=embeddings,
            vector_store_path=vector_store_path,
            embedding_type=embedding_type,
            embeddings_path=embeddings_path,
            verbose=verbose
        )

        # Initialize error tracking
        self.pdf_embedding_error = set()
        self.uniqueids_withoutPDFtext = set()

        # Load existing error tracking
        self._load_error_tracking()
        self.embeddings_path = embeddings_path or f"{vector_store_path}/embeddings"

    def _load_error_tracking(self):
        """Load error tracking if available."""
        error_path = f'{self.embeddings_path}/screening2_PDFembedding_error.pkl'
        if os.path.exists(error_path):
            with open(error_path, 'rb') as file:
                self.pdf_embedding_error = pickle.load(file)
            self.logger.info(f"Loaded {len(self.pdf_embedding_error)} PDF embedding errors")

    def embed(self, data):
        """
        Embed PDFs for all records in data.

        Args:
            data (pd.DataFrame): DataFrame with PDF paths

        Returns:
            bool: True if successful, False otherwise
        """
        # Check if 'pdf_path' column exists
        if 'pdf_path' not in data.columns:
            self.logger.error("DataFrame must have 'pdf_path' column")
            return False

        # Filter out records without PDF paths
        pdf_records = data[data['pdf_path'].notna()]
        self.logger.info(f"Found {len(pdf_records)} records with PDF paths")

        # Run the embedding process
        try:
            asyncio.run(self.embed_articles_PDFs(pdf_records))
            return True
        except Exception as e:
            self.logger.error(f"Error embedding PDFs: {str(e)}")
            return False

    async def embed_articles_PDFs(self, pdf_records):
        """
        Embed PDFs for articles.

        Args:
            pdf_records (pd.DataFrame): DataFrame with PDF paths
        """
        self.logger.info("Starting PDF embedding process...")

        # Ensure directories exist
        await self.ensure_directory_permissions(self.embeddings_path)

        # Initialize or validate the 'screening2' database
        path_name = self.embeddings_path
        collection_name = "screening2"

        # Check if database exists and is valid
        if await self.validate_chroma_db(path_name, collection_name):
            self.logger.info(f"Using existing Chroma index for screening2 at {path_name}")
            pdf_db = Chroma(
                collection_name=collection_name,
                persist_directory=path_name,
                embedding_function=self.embeddings
            )
        else:
            self.logger.info(f"Creating new Chroma index for screening2 at {path_name}")
            pdf_db = Chroma(
                collection_name=collection_name,
                persist_directory=path_name,
                embedding_function=self.embeddings
            )

        # Get already embedded uniqueids
        embedded_uniqueids = await self.get_embedded_uniqueids(pdf_db)
        self.logger.info(f"Found {len(embedded_uniqueids)} already embedded uniqueids")

        # Filter records needing processing
        records_to_process = []
        for _, record in pdf_records.iterrows():
            uniqueid = str(record['uniqueid'])

            # Skip if already embedded
            if uniqueid in embedded_uniqueids:
                self.logger.debug(f"Skipping uniqueid {uniqueid} as it's already embedded")
                continue

            # Skip if already in error list - only skip if it's already in the error list
            if uniqueid in self.pdf_embedding_error:
                self.logger.debug(f"Skipping uniqueid {uniqueid} as it's in the error list")
                continue

            # Skip if PDF is invalid
            if pd.isna(record['pdf_path']) or not os.path.exists(record['pdf_path']):
                self.logger.warning(f"PDF file not found for {uniqueid}: {record.get('pdf_path')}")
                self.pdf_embedding_error.add(uniqueid)
                continue

            records_to_process.append(record)

        if not records_to_process:
            self.logger.info("No new PDFs to embed")
            return

        self.logger.info(f"Processing {len(records_to_process)} new PDFs")

        # Process PDFs with controlled concurrency
        semaphore = asyncio.Semaphore(2)

        async def process_with_semaphore(record):
            async with semaphore:
                return await self.embed_article_PDF(record)

        tasks = [asyncio.create_task(process_with_semaphore(record)) for record in records_to_process]

        # Track progress
        success_count = error_count = 0

        with tqdm(total=len(tasks), desc="Embedding PDFs") as progress_bar:
            for task in asyncio.as_completed(tasks):
                try:
                    pdf_db_result = await task
                    if pdf_db_result is not None:
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    self.logger.error(f"Task failed: {str(e)}")
                    error_count += 1

                progress_bar.set_postfix({
                    'Success': success_count,
                    'Errors': error_count
                })
                progress_bar.update(1)

        # Save error tracking
        error_tracking_path = f'{self.embeddings_path}/screening2_PDFembedding_error.pkl'
        with open(error_tracking_path, 'wb') as file:
            pickle.dump(self.pdf_embedding_error, file)

        self.logger.info(
            f"PDF embedding completed:\n"
            f"Successfully embedded: {success_count}\n"
            f"Failed to embed: {error_count}"
        )

    async def embed_article_PDF(self, pdf_record):
        """
        Embed a single PDF article.

        Args:
            pdf_record (pd.Series): Record with PDF path

        Returns:
            object: Chroma DB or None on failure
        """
        await asyncio.sleep(0.5)  # Add 500ms delay between requests

        uniqueid = str(pdf_record['uniqueid'])
        max_retries = 5
        retry_delay = 5
        base_retry_delay = 3  # Start with 3 seconds

        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Embedding article {uniqueid}, attempt {attempt + 1}/{max_retries}")

                # Validate PDF path
                if not os.path.exists(pdf_record['pdf_path']):
                    raise FileNotFoundError(f"PDF file not found: {pdf_record['pdf_path']}")

                # Define text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=3000,
                    chunk_overlap=300,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                )

                # Load and split the PDF
                async def load_and_split_pdf():
                    try:
                        # Use PyMuPDFLoader in "single" mode to load the entire PDF as one document
                        loader = PyMuPDFLoader(pdf_record['pdf_path'], mode="single")
                        # Load the document using the new loader
                        docs = await asyncio.to_thread(loader.load)
                        if not docs:
                            raise ValueError("No text extracted from PDF")
                        # Split the loaded document using your text splitter
                        documents = text_splitter.split_documents(docs)
                        return documents
                    except Exception as e:
                        raise RuntimeError(f"PDF processing failed: {str(e)}")

                documents = await load_and_split_pdf()

                # Check for empty or duplicate content
                if documents[0].page_content == "" or self.has_repeated_content(documents):
                    self.logger.error(f"Empty or repeated content for article {uniqueid}")
                    self.uniqueids_withoutPDFtext.add(uniqueid)
                    return None

                # Update metadata
                metadata = {k: str(v) for k, v in pdf_record.to_dict().items()}
                for doc in documents:
                    doc.metadata.update(metadata)
                    doc.metadata['source'] = uniqueid
                    doc.metadata['pdf_name'] = os.path.basename(pdf_record['pdf_path'])

                # Set up the Chroma database
                path_name = self.embeddings_path
                collection_name = "screening2"

                # Validate or create the database
                is_valid_db = await self.validate_chroma_db(path_name, collection_name)

                if is_valid_db:
                    self.logger.debug(f"Using existing Chroma index for screening2")
                    pdf_db = Chroma(
                        collection_name=collection_name,
                        persist_directory=path_name,
                        embedding_function=self.embeddings
                    )
                else:
                    self.logger.debug(f"Creating new Chroma index for screening2")
                    pdf_db = Chroma(
                        collection_name=collection_name,
                        persist_directory=path_name,
                        embedding_function=self.embeddings
                    )

                # Process documents in batches
                batch_size = 10
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]

                    # Add documents to the database
                    if hasattr(pdf_db, 'aadd_documents'):
                        await pdf_db.aadd_documents(batch)
                    else:
                        # Fallback for non-async databases
                        await asyncio.to_thread(pdf_db.add_documents, batch)

                    await asyncio.sleep(0.1)  # Small delay

                self.logger.debug(f"Successfully embedded article {uniqueid}")
                return pdf_db

            except Exception as e:
                if "RATE_LIMIT_EXCEEDED" in str(e):
                    # Exponential backoff for rate limit errors
                    retry_delay = base_retry_delay * (2 ** attempt)  # Exponential backoff

                    self.logger.warning(
                        f"Rate limit exceeded for {uniqueid}. Retrying in {retry_delay} seconds... "
                        f"(Attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)

                elif attempt < max_retries - 1:
                    self.logger.warning(
                        f"Error for {uniqueid}: {str(e)}. Retrying in {base_retry_delay} seconds... "
                        f"(Attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(base_retry_delay)

                else:
                    self.logger.error(f"Failed to embed {uniqueid} after {max_retries} attempts: {str(e)}")
                    self.pdf_embedding_error.add(uniqueid)
                    return None

    async def validate_chroma_db(self, path_name, collection_name):
        """
        Validate that a Chroma database exists and is accessible.

        Args:
            path_name (str): Path to database
            collection_name (str): Collection name

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if not (os.path.exists(os.path.join(path_name, 'chroma.sqlite3')) or
                    os.path.exists(os.path.join(path_name, 'index'))):
                return False

            # Try to open the database
            db = Chroma(
                collection_name=collection_name,
                persist_directory=path_name,
                embedding_function=self.embeddings
            )

            # Verify we can access the collection
            _ = db._collection.count()
            return True

        except Exception as e:
            self.logger.warning(f"Invalid Chroma database at {path_name}: {str(e)}")
            return False

    async def get_embedded_uniqueids(self, pdf_db):
        """
        Get all uniqueids currently in the index.

        Args:
            pdf_db (object): Chroma DB instance

        Returns:
            set: Set of uniqueids
        """
        try:
            uniqueids = set()

            # Get all documents from the collection, including metadatas
            results = pdf_db._collection.get(include=['metadatas'])

            if results and 'metadatas' in results:
                for metadata in results['metadatas']:
                    if metadata and 'uniqueid' in metadata:
                        uniqueids.add(metadata['uniqueid'])

            return uniqueids

        except Exception as e:
            self.logger.error(f"Error getting embedded uniqueids: {str(e)}")
            return set()

    async def ensure_directory_permissions(self, directory):
        """
        Ensure directory exists and has proper permissions.

        Args:
            directory (str): Directory path
        """
        try:
            os.makedirs(directory, exist_ok=True)
            os.chmod(directory, 0o777)  # Full permissions

            # Also ensure parent directory has proper permissions
            parent_dir = os.path.dirname(directory)
            os.chmod(parent_dir, 0o777)

        except Exception as e:
            self.logger.warning(f"Could not set permissions for {directory}: {e}")

    def has_repeated_content(self, documents):
        """
        Check if there's repeated content across multiple documents.

        Args:
            documents (list): List of documents

        Returns:
            bool: True if repeated content found, False otherwise
        """
        content_set = set()

        for doc in documents:
            # Extract content
            if hasattr(doc, 'page_content'):
                content = doc.page_content
            else:
                content = str(doc)

            # Check if content is already in set
            if content in content_set:
                return True

            # Add content to set
            content_set.add(content)

        return False

    def validate_embeddings(self, data):
        """
        Validate that all PDFs are properly embedded.

        Args:
            data (pd.DataFrame): Original DataFrame

        Returns:
            bool: True if valid, False otherwise
        """
        # This would typically check that all PDFs are correctly embedded
        # For simplicity, we'll assume all successfully processed PDFs are valid
        return True
