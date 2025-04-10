import os.path
import pickle
import sqlite3
import threading
from langchain_community.document_loaders import DataFrameLoader
import os
from tqdm import tqdm
import pandas as pd
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.prompts import ChatPromptTemplate
import json
from dataset_visualizer import DatasetVisualizer
import logging
import asyncio
from tqdm.asyncio import tqdm_asyncio
from langchain_chroma import Chroma
import tempfile
import shutil
from pdf_downloading_module.PDF_downloader import ArticleDownloader
import hashlib
import re
from chatbot.selfrag import ChatbotApp, clean_answer, remap_source_citations


email = 'fernando.miguel.delgado-chaves@uni-hamburg.de'
# Set logging level to suppress DEBUG messages from other libraries
logging.getLogger("chromadb").setLevel(logging.INFO)
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)


class academate:
    def __init__(self, topic: str, llm: object, embeddings: object, criteria_dict: dict, vector_store_path: str,
                 literature_df: pd.DataFrame = None, content_column: str = "record", embeddings_path: str = None,
                 pdf_location: str = None, verbose: bool = False, chunksize: int = 25) -> None:

        self.embeddings_path = embeddings_path
        self.visualizer = DatasetVisualizer(llm)
        self.verbose = verbose  # Store the verbose flag

        # Configure your logger
        self.logger = logging.getLogger(__name__)

        # Set logging level based on verbose flag
        self.logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        # Only add handlers if they are not already added
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False  # Prevent duplicate logs

        # Suppress DEBUG messages from pdfminer
        logging.getLogger('pdfminer').setLevel(logging.WARNING)

        if not self.verbose:
            # Suppress DEBUG messages from httpcore and httpx
            logging.getLogger('httpcore').setLevel(logging.WARNING)
            logging.getLogger('httpx').setLevel(logging.WARNING)

        # Initial attributes
        self.topic: str = topic
        self.criteria_dict = criteria_dict
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store_path = vector_store_path
        self.literature_df = literature_df

        if self.embeddings_path is None:
            self.embeddings_path = f"{self.vector_store_path}/embeddings"
        self.content_column = content_column

        if self.literature_df is not None:
            self.literature_df.reset_index(drop=True, inplace=True)
            self.literature_df['uniqueid'] = self.literature_df.apply(self.generate_uniqueid, axis=1)
            self.check_and_remove_duplicates()

        # Screening attributes
        self.selected_fetch_k = 20
        self.selected_k = len(self.criteria_dict)

        # Screening 1 attributes
        self.results_screening1 = None
        self.screening1_record2answer = dict()
        self.screening1_missing_records = set()
        self.uniqueids_withoutPDFtext = set()
        self.chunksize = chunksize
        self.vectorDB_screening1 = None

        self.screening1_dir = f"{self.vector_store_path}/screening1"
        if not os.path.exists(self.screening1_dir):
            os.makedirs(self.screening1_dir)
            os.chmod(self.screening1_dir, 0o777)  # Grant all permissions

        self.embeddings_path1 = f"{self.embeddings_path}/screening1_embeddings"
        if not os.path.exists(self.embeddings_path1):
            os.makedirs(self.embeddings_path1)
            os.chmod(self.embeddings_path1, 0o777)  # Grant all permissions

        # Screening 2 attributes
        self.results_screening2 = None
        self.screening2_record2answer = dict()
        self.screening2_PDFdownload_error = set()
        self.screening2_PDFembedding_error = set()
        self.screening2_missing_records = set()
        self.vectorDB_screening2 = None
        self.screening2_dir = f"{self.vector_store_path}/screening2"
        if not os.path.exists(self.screening2_dir):
            os.makedirs(self.screening2_dir)
            os.chmod(self.screening2_dir, 0o777)  # Grant all permissions

        self.embeddings_path2 = f"{self.embeddings_path}/screening2_embeddings"
        if not os.path.exists(self.embeddings_path2):
            os.makedirs(self.embeddings_path2)
            os.chmod(self.embeddings_path2, 0o777)  # Grant all permissions

        if pdf_location is None:
            self.pdf_location = f"{self.vector_store_path}/pdfs"
        else:
            self.pdf_location = pdf_location
        if not os.path.exists(self.pdf_location):
            os.makedirs(self.pdf_location)
            os.chmod(self.pdf_location, 0o777)

        # Initialize ArticleDownloader
        self.article_downloader = ArticleDownloader(
            output_directory=self.pdf_location,
            email=email or "your.email@example.com",
            logger=self.logger,
            verbose=verbose
        )

        # chatting
        self.db_qa = None
        self.analysis_time = None
        self.db = None

    def check_and_remove_duplicates(self):
        """
        Checks for duplicate unique IDs in the DataFrame, warns the user,
        and removes duplicates based on the 'Record' column.
        """
        num_duplicates = len(self.literature_df) - len(set(self.literature_df['uniqueid']))

        if num_duplicates > 0:
            self.logger.warning(f"Detected {num_duplicates} duplicate unique IDs in the DataFrame.")

            # Find and print duplicate rows based on 'uniqueid'
            duplicate_rows = self.literature_df[self.literature_df.duplicated('uniqueid', keep=False)]
            duplicate_rows = duplicate_rows.sort_values(by='uniqueid')
            self.logger.warning("Duplicate rows based on uniqueid:")
            for index, row in duplicate_rows.iterrows():
                self.logger.warning(f"  uniqueid: {row['uniqueid']}, Record: {row['Record']}")

            # Remove duplicates based on 'Record' column
            self.literature_df.drop_duplicates(subset='uniqueid', keep='first', inplace=True)
            self.logger.info(f"Removed duplicates based on '{self.content_column}' column. {len(self.literature_df)} rows remaining.")
        else:
            self.logger.info("No duplicate unique IDs found.")

    def generate_uniqueid(self, row):
        """
        Generates a consistent row ID based on the normalized description.
        Combines normalization and hashing into a single function.
        """
        # Normalize while preserving some punctuation
        normalized_description = re.sub(r'[^a-zA-Z0-9 \-():]', '', row[self.content_column])

        # Normalize whitespace
        normalized_description = ' '.join(normalized_description.split())

        key_string = f"{normalized_description}"
        id_record = hashlib.sha256(key_string.encode()).hexdigest()[:20]
        return id_record

    # EMBEDDINGS
    def embed_literature_df(self, path_name=None):
        if path_name is None:
            path_name = self.embeddings_path1
        else:
            os.makedirs(path_name, exist_ok=True)
        if self.verbose:
            print("Number of records: ", len(self.literature_df))
        loader = DataFrameLoader(self.literature_df, page_content_column=self.content_column)

        # Load all documents
        all_docs = loader.load()

        # Ensure 'uniqueid' is in metadata and is a string
        for doc in all_docs:
            if 'uniqueid' in doc.metadata:
                doc.metadata['uniqueid'] = str(doc.metadata['uniqueid'])
            else:
                self.logger.warning(f"'uniqueid' not found in metadata for doc: {doc}")

        # Check if Chroma collection exists
        if os.path.exists(os.path.join(path_name, 'chroma.sqlite3')) or \
                os.path.exists(os.path.join(path_name, 'index')):
            if self.verbose:
                print('Loading existing Chroma index...')
            self.db = Chroma(
                collection_name="literature",
                persist_directory=path_name,
                embedding_function=self.embeddings
            )
        else:
            if self.verbose:
                print('No existing Chroma index found. Creating a new one...')
            # Create the Chroma index
            self.db = Chroma(
                collection_name="literature",
                persist_directory=path_name,
                embedding_function=self.embeddings
            )
            # Embed documents in batches
            batch_size = 100  # Adjust as needed
            total_batches = (len(all_docs) + batch_size - 1) // batch_size
            with tqdm(total=total_batches, desc="Embedding documents") as progress_bar:
                for i in range(0, len(all_docs), batch_size):
                    batch_docs = all_docs[i:i + batch_size]
                    texts = [doc.page_content for doc in batch_docs]
                    metadatas = [doc.metadata for doc in batch_docs]
                    # Ensure all metadata values are strings
                    metadatas = [{k: str(v) for k, v in metadata.items()} for metadata in metadatas]
                    self.db.add_texts(texts, metadatas=metadatas)
                    progress_bar.update(1)

        # Validate the index
        self.validate_index()
        return self.db

    # PROMPTING AND CHAINS

    def parse_json_safely(self, json_string):
        import re
        import json
        from json import JSONDecodeError

        # Use regex to extract the JSON object
        json_matches = re.findall(r'\{.*\}', json_string, re.DOTALL)
        if json_matches:
            json_data = json_matches[0]
            try:
                return json.loads(json_data)
            except JSONDecodeError as e:
                # Attempt to fix the JSON
                fixed_json_data = self.fix_json(json_data)
                try:
                    return json.loads(fixed_json_data)
                except JSONDecodeError as e:
                    self.logger.error(
                        f"Failed to parse JSON after attempting to fix: {e}\nOriginal Response was: {json_string}")
                    return {}
        else:
            self.logger.error(f"No JSON found in the response: {json_string}")
            return {}

    def fix_json(self, json_string):
        # Implement simple fixes, such as adding missing closing braces
        # or quotes. For more complex fixes, consider using a library like 'json5' or 'demjson'

        # Example: Add missing closing braces
        open_braces = json_string.count('{')
        close_braces = json_string.count('}')
        json_string += '}' * (open_braces - close_braces)

        # Example: Close unclosed quotes
        open_quotes = json_string.count('"') % 2
        if open_quotes:
            json_string += '"'

        return json_string

    def prepare_screening_prompt(self):
        # Build the formatted criteria and JSON structure from the criteria_dict
        formatted_criteria = "\n".join(f"- {key}: {value}" for key, value in self.criteria_dict.items())
        json_structure = json.dumps(
            {key: {"label": "boolean", "reason": "string"} for key in self.criteria_dict.keys()},
            indent=2)

        prompt = ChatPromptTemplate.from_template("""
            Analyze the following scientific article and determine if it meets the specified criteria.
            Only use the information provided in the context.

            Context: {context}

            Criteria:
            {criteria}

            For each criterion, provide a boolean label (true if it meets the criterion, false if it doesn't)
            and a brief reason (no more than 20 words) for your decision.

            **Important**: Respond **only** with a JSON object matching the following structure, and do not include any additional text:

            {json_structure}

            Ensure your response is a valid JSON object.
            """)

        return prompt, formatted_criteria, json_structure

    def prepare_chain(self):
        chain = (
                ChatPromptTemplate.from_template("""
                Analyze the following scientific article and determine if it meets the specified criteria.
                Only use the information provided in the context.

                Context: {context}

                Criteria:
                {criteria}

                For each criterion, provide a boolean label (true if it meets the criterion, false if it doesn't)
                and a brief reason for your decision.

                **Important**: Respond **only** with a JSON object matching the following structure, and do not include any additional text:

                {json_structure}

                Ensure your response is a valid JSON object.
                """)
                | self.llm
                | (lambda x: self.parse_json_safely(x.content))
        )
        return chain

    def get_indexed_uniqueids(self):
        """Get all uniqueids currently in the Chroma index."""
        uniqueids = set()
        # Get all documents from the collection, including metadatas
        results = self.db._collection.get(include=['metadatas'])
        if results and 'metadatas' in results:
            for metadata in results['metadatas']:
                if metadata and 'uniqueid' in metadata:
                    uniqueids.add(metadata['uniqueid'])
        return uniqueids

    def validate_index(self):
        indexed_uniqueids = self.get_indexed_uniqueids()
        df_uniqueids = set(self.literature_df['uniqueid'].astype(str))
        missing_uniqueids = df_uniqueids - indexed_uniqueids
        extra_uniqueids = indexed_uniqueids - df_uniqueids

        if missing_uniqueids:
            self.logger.warning(f"Missing {len(missing_uniqueids)} documents in the index.")
            self.logger.debug(f"Missing uniqueids: {missing_uniqueids}")
        else:
            self.logger.info("All documents are correctly embedded in the index.")

        if extra_uniqueids:
            self.logger.warning(
                f"There are {len(extra_uniqueids)} extra documents in the index not present in the dataframe.")
            self.logger.debug(f"Extra uniqueids: {extra_uniqueids}")

        # BASE SCREENING

    @staticmethod
    def atomic_save(file_path, data):
        """
        Saves data to a file atomically.
        """
        dir_name = os.path.dirname(file_path)
        with tempfile.NamedTemporaryFile('wb', delete=False, dir=dir_name) as tmp_file:
            pickle.dump(data, tmp_file)
            temp_name = tmp_file.name
        shutil.move(temp_name, file_path)

    @staticmethod
    def has_repeated_content(documents):
        """
        Check if there's repeated content across multiple documents.
        Args:
            documents: A list of document objects with a 'page_content' attribute or a list of strings
        Returns:
            bool: True if any identical content appears more than once, False otherwise
        """
        content_set = set()
        for doc in documents:
            # Extract content (handle both document objects with page_content attribute and direct strings)
            if hasattr(doc, 'page_content'):
                content = doc.page_content
            else:
                content = str(doc)

            # If we've seen this content before, we found a duplicate
            if content in content_set:
                return True

            # Add the content to our set of seen content
            content_set.add(content)

        # No duplicates found
        return False

    def save_screening_results(self, screening_type):
        with threading.Lock():
            screening_dir = getattr(self, f"{screening_type}_dir")
            record2answer = getattr(self, f"{screening_type}_record2answer")
            missing_records = getattr(self, f"{screening_type}_missing_records")

            # Save atomically
            self.atomic_save(f'{screening_dir}/{screening_type}_predicted_criteria.pkl', record2answer)
            self.atomic_save(f'{screening_dir}/{screening_type}_missing_records.pkl', missing_records)

    def validate_mutual_exclusivity(self, screening_type):
        """
        Validates that no record exists in both record2answer and missing_records.
        """
        record2answer_attr = f"{screening_type}_record2answer"
        missing_records_attr = f"{screening_type}_missing_records"

        record2answer = getattr(self, record2answer_attr)
        missing_records = getattr(self, missing_records_attr)

        overlapping_records = set(record2answer.keys()).intersection(missing_records)
        if overlapping_records:
            self.logger.error(f"Mutual exclusivity violated! Overlapping records: {overlapping_records}")
            # Resolve overlaps by removing from missing_records
            missing_records -= overlapping_records
            self.logger.info(f"Removed overlapping records from missing_records: {overlapping_records}")
            # Save corrected data
            self.save_screening_results(screening_type)
        else:
            self.logger.info("Mutual exclusivity validated successfully.")

    def load_existing_screening_data(self, screening_type):
        screening_dir = getattr(self, f"{screening_type}_dir")
        record2answer_attr = f"{screening_type}_record2answer"
        missing_records_attr = f"{screening_type}_missing_records"

        record2answer_path = f'{screening_dir}/{screening_type}_predicted_criteria.pkl'
        missing_records_path = f'{screening_dir}/{screening_type}_missing_records.pkl'

        if os.path.exists(record2answer_path):
            with open(record2answer_path, 'rb') as file:
                setattr(self, record2answer_attr, pickle.load(file))
        else:
            setattr(self, record2answer_attr, {})

        # with open(record2answer_path, 'rb') as f:
        #     screening2_record2answer = pickle.load(f)

        if os.path.exists(missing_records_path):
            with open(missing_records_path, 'rb') as file:
                setattr(self, missing_records_attr, pickle.load(file))
        else:
            setattr(self, missing_records_attr, set())

        self.logger.info(
            f"Loaded {len(getattr(self, record2answer_attr))} existing records for {screening_type}\nLoaded {len(getattr(self, missing_records_attr))} missing records for {screening_type}")
        return getattr(self, record2answer_attr), getattr(self, missing_records_attr)

    def get_rec_numbers(self, screening_type):
        if screening_type == 'screening1':
            return self.literature_df['uniqueid'].astype(str).to_list()
        elif screening_type == 'screening2':
            return self.results_screening2['uniqueid'].astype(str).to_list()
        else:
            raise ValueError(f"Invalid screening type: {screening_type}")

    async def process_single_record_async(self, recnumber, chain, screening_type):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if screening_type == 'screening1':
                    # Fetch the document directly
                    document_row = self.literature_df[self.literature_df['uniqueid'] == recnumber]
                    if document_row.empty:
                        self.logger.error(f"No document found for record {recnumber}")
                        return recnumber, None
                    context = document_row[self.content_column].iloc[0]

                elif screening_type == 'screening2':
                    # screening2
                    # Load the Chroma index for this PDF
                    # recnumber = records_to_process[0]
                    path_name = f"{self.embeddings_path2}/pdf_{recnumber}"
                    if os.path.exists(os.path.join(path_name, 'chroma.sqlite3')) or \
                            os.path.exists(os.path.join(path_name, 'index')):
                        pdf_db = Chroma(
                            collection_name=f"pdf_{recnumber}",
                            persist_directory=path_name,
                            embedding_function=self.embeddings
                        )
                    else:
                        self.logger.error(f"No Chroma index found for PDF {recnumber}.")
                        return recnumber, None

                    # Retrieve relevant chunks per criterion
                    all_retrieved_docs = []
                    for key, value in self.criteria_dict.items():
                        # Create a retriever and retrieve top k relevant documents
                        retriever = pdf_db.as_retriever(
                            search_type='mmr',  # Use 'mmr' as per your preference
                            search_kwargs={'k': 2, "score_threshold": 0.0}  # Adjust k as needed
                        )
                        retrieved_docs = retriever.invoke(value)

                        if not retrieved_docs:
                            self.logger.warning(f"No documents retrieved for criterion '{key}' in record {recnumber}")
                            collection = pdf_db._client.get_collection(f"pdf_{recnumber}")
                            num_documents = collection.count()
                            self.logger.warning(f"Number of documents in the collection: {num_documents}")
                            continue  # Decide whether to continue or handle differently

                        all_retrieved_docs.extend(retrieved_docs)

                    if not all_retrieved_docs:
                        self.logger.error(f"No documents retrieved for any criteria in record {recnumber}")
                        return recnumber, None

                    # Remove duplicate documents based on page content
                    unique_docs = {doc.page_content: doc for doc in all_retrieved_docs}.values()

                    # Extract page content from unique retrieved documents
                    context = "\n\n".join([doc.page_content for doc in unique_docs])

                else:
                    raise ValueError(f"Invalid screening type: {screening_type}")

                inputs = {
                    "context": context,
                    "criteria": self.formatted_criteria,
                    "json_structure": self.json_structure
                }

                # Invoke the chain asynchronously
                result = await chain.ainvoke(inputs)

                if result:
                    return recnumber, result
                else:
                    self.logger.warning(f"Unexpected response format for record {recnumber}. Retrying...")

            except Exception as e:
                if "Event loop is closed" in str(e):
                    # If the event loop is closed, no point in retrying with the same loop
                    self.logger.error(f"Event loop closed for record {recnumber}. Cannot continue.")
                    return recnumber, None
                self.logger.error(f"Error processing record {recnumber}: {str(e)}")

                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying record {recnumber}... (Attempt {attempt + 2}/{max_retries})")
                    await asyncio.sleep(2)

                else:
                    self.logger.error(f"Failed to process record {recnumber} after {max_retries} attempts.")
                    break

        return recnumber, None

    async def process_records_concurrently(self, records_to_process, chain, screening_type):
        """
        Process records concurrently with improved semaphore handling
        """
        # Initialize counters
        success_count = 0
        failure_count = 0

        # Get the appropriate record tracking attributes
        record2answer_attr = f"{screening_type}_record2answer"
        missing_records_attr = f"{screening_type}_missing_records"
        record2answer = getattr(self, record2answer_attr)
        missing_records = getattr(self, missing_records_attr)

        # Create semaphore in the async context
        sem = asyncio.Semaphore(5)
        # recnumber = records_to_process[0]

        async def process_with_semaphore(recnumber):
            async with sem:
                return await self.process_single_record_async(recnumber, chain, screening_type)

        # Create tasks for each record
        tasks = []
        for recnumber in records_to_process:
            # print(recnumber)
            # task=recnumber
            task = asyncio.create_task(process_with_semaphore(recnumber))
            tasks.append(task)

        # Process tasks with progress tracking
        results = []
        with tqdm(total=len(tasks), desc=f"Processing Records ({screening_type})") as pbar:
            for coro in asyncio.as_completed(tasks):
                try:
                    recnumber, result = await coro
                    results.append((recnumber, result))

                    if result:
                        record2answer[recnumber] = result
                        missing_records.discard(recnumber)
                        success_count += 1
                    else:
                        missing_records.add(recnumber)
                        failure_count += 1

                    # Update progress bar
                    pbar.set_postfix({
                        'Success': success_count,
                        'Failure': failure_count
                    })
                    pbar.update(1)

                    # Periodically save results
                    if (success_count + failure_count) % 20 == 0:
                        self.save_screening_results(screening_type)

                except Exception as e:
                    self.logger.error(f"Task processing error: {str(e)}")
                    failure_count += 1
                    pbar.update(1)

        return results

    def structure_output(self, answerset: dict = None):
        data_dict = {}
        missing_keys = set()
        for key in answerset.keys():
            if not isinstance(answerset[key], dict):
                self.logger.error(f"Unexpected type for record {key}: {type(answerset[key])}. Skipping this record.")
                missing_keys.add(key)
                continue
            data_dict[key] = {}
            for checkpoint_studied in self.criteria_dict.keys():
                answer = answerset[key].get(checkpoint_studied)
                if isinstance(answer, dict):
                    data_dict[key][checkpoint_studied] = answer.get('label', False)
                elif isinstance(answer, bool):  # Handle boolean directly
                    data_dict[key][checkpoint_studied] = answer
                else:
                    self.logger.warning(
                        f"Unexpected type for criterion '{checkpoint_studied}' in record {key}: {type(answer)}")
                    data_dict[key][checkpoint_studied] = False
                    missing_keys.add(checkpoint_studied)

        if missing_keys and self.verbose:
            print(f"Warning: The following keys were not found in some records: {', '.join(missing_keys)}")

        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df['uniqueid'] = df.index.astype(str)
        df.reset_index(inplace=True, drop=True)
        return df, missing_keys

    def base_screening(self, screening_type):
        # Prepare prompt and other settings
        self.prompt, self.formatted_criteria, self.json_structure = self.prepare_screening_prompt()

        # Load existing screening data
        self.load_existing_screening_data(screening_type=screening_type)
        record2answer_attr = f"{screening_type}_record2answer"
        missing_records_attr = f"{screening_type}_missing_records"

        record2answer = getattr(self, record2answer_attr)
        missing_records = getattr(self, missing_records_attr)

        # --- Filter uniqueids based on current literature_df ---
        current_uniqueids = set(self.get_rec_numbers(screening_type=screening_type))
        record2answer = {k: v for k, v in record2answer.items() if k in current_uniqueids}
        missing_records = {rec for rec in missing_records if rec in current_uniqueids}
        setattr(self, record2answer_attr, record2answer)
        setattr(self, missing_records_attr, missing_records)

        # Determine records to process
        records_to_process = [rec for rec in self.get_rec_numbers(screening_type=screening_type) if rec not in record2answer]

        # Add missing records to records_to_process
        for rec in missing_records:
            if rec not in records_to_process:
                records_to_process.append(rec)

        # If there are no records to process, exit early
        if not records_to_process:
            self.logger.info("All records have been processed, or only previously processed records are present.")
            return

        # Create the chain
        chain = self.prepare_chain()

        # Run the asynchronous processing
        results = asyncio.run(self.process_records_concurrently(records_to_process=records_to_process, chain=chain, screening_type=screening_type))

        # Display final summary
        total_records = len(records_to_process)
        success_count = len([r for r in results if r[1]])
        failure_count = len([r for r in results if not r[1]])

        self.logger.info(f"\nTotal records processed: {total_records}")
        self.logger.info(f"Successfully processed: {success_count}")
        self.logger.info(f"Failed to process: {failure_count}")

        # Save final results
        self.save_screening_results(screening_type)

    def select_articles_based_on_criteria(self, df: pd.DataFrame = None):
        self.logger.info(f"Selecting articles based on criteria...")
        for column in self.criteria_dict.keys():
            df[column] = df[column].astype(bool)

        columns_needed_as_true = list(self.criteria_dict.keys())
        df['predicted_screening'] = df[columns_needed_as_true].all(axis=1)
        return df

    # SCREENING 1 and 2
    def run_screening1(self):
        self.logger.info(f"Total number of records: {len(self.literature_df)}")

        start_time = time.time()
        self.base_screening(screening_type='screening1')
        # Validate mutual exclusivity
        self.validate_mutual_exclusivity(screening_type='screening1')
        self.analysis_time = time.time() - start_time
        self.results_screening1 = self.post_screening_analysis(screening_type='screening1')
        self.results_screening1['runtime_scr1'] = self.analysis_time
        # rename criteria columns to add '_scr1'
        self.results_screening1.rename(columns={key: key + '_scr1' for key in self.criteria_dict.keys()}, inplace=True)

        # Add this line to rename 'predicted_screening' to 'predicted_screening1'
        self.results_screening1.rename(columns={'predicted_screening': 'predicted_screening1'}, inplace=True)

        # self.logger.info(f"Columns
        # self.logger.info(f"Columns in results_screening1: {self.results_screening1.columns}")

        return self.results_screening1

    def run_screening2(self):
        if self.results_screening1 is None:
            if 'pdf_path' not in self.literature_df.columns:
                raise ValueError(
                    "Either run_screening1() must be called before run_screening2() or 'pdf_path' column must be provided in the initial DataFrame.")
            else:
                self.logger.info("Using provided PDF paths for screening2.")
                self.results_screening2 = self.literature_df.copy()
        else:
            if 'predicted_screening1' not in self.results_screening1.columns:
                self.logger.warning("Warning: 'predicted_screening1' column not found. Using all records for screening2.")
                self.results_screening2 = self.results_screening1.copy()
            else:
                self.results_screening2 = self.results_screening1[
                    self.results_screening1['predicted_screening1'] == True]

        self.logger.info(f"Records selected for PDF processing: {len(self.results_screening2)}")
        self.logger.debug("DEBUG: results_screening2 before embed_articles_PDFs:")
        self.logger.debug(self.results_screening2[['uniqueid', 'pdf_path']])
        # print("DEBUG: Unique IDs in results_screening2:", self.results_screening2['uniqueid'].unique())
        self.logger.debug(f"Number of unique IDs: {len(self.results_screening2['uniqueid'].unique())}")

        # If PDFs are not already downloaded, download them
        if 'pdf_path' not in self.results_screening2.columns or self.results_screening2['pdf_path'].isnull().any():
            # Download PDFs and update the DataFrame
            self.results_screening2 = self.download_documents()  # Add this line to update results_screening2

            if len(self.results_screening2[self.results_screening2['pdf_path'].notna()]) > 0:
                asyncio.run(self.embed_articles_PDFs())
        else:
            asyncio.run(self.embed_articles_PDFs())

        # Filter out records where PDF is not available

        self.results_screening2 = self.results_screening2[~self.results_screening2['uniqueid'].isin(self.screening2_PDFembedding_error)]

        self.logger.info(f"Records with PDF proceeding for screening2: {len(self.results_screening2)}")

        if len(self.results_screening2) > 0:
            start_time = time.time()
            self.base_screening(screening_type='screening2')
            self.validate_mutual_exclusivity('screening2')
            self.analysis_time = time.time() - start_time
            self.results_screening2 = self.post_screening_analysis('screening2')
            self.results_screening2['runtime_scr2'] = self.analysis_time
            self.results_screening2.rename(columns={key: key + '_scr2' for key in self.criteria_dict.keys()},
                                           inplace=True)
            self.results_screening2.rename(columns={'predicted_screening': 'predicted_screening2'}, inplace=True)

        else:
            print("No PDFs were successfully processed. Skipping screening2.")

        return self.results_screening2

    def post_screening_analysis(self, screening_type):
        record2answer = getattr(self, f"{screening_type}_record2answer")
        missing_records = getattr(self, f"{screening_type}_missing_records")

        results_attr = f"results_{screening_type}"
        if hasattr(self, results_attr) and getattr(self, results_attr) is not None:
            total_records = len(getattr(self, results_attr))
        else:
            total_records = len(self.literature_df)  # Fallback to initial dataset

        correctly_analyzed = len(record2answer)
        incorrectly_analyzed = max(0, total_records - correctly_analyzed)  # Ensure this is not negative

        self.logger.info(
            f"Total records for {screening_type}: {total_records}\nCORRECTLY analyzed: {correctly_analyzed}\nINCORRECTLY analyzed: {incorrectly_analyzed}")

        inconsistent_records, extra_keys, missing_keys = self.analyze_model_output(screening_type)

        if inconsistent_records or extra_keys or missing_keys:
            print("There are inconsistencies between your criteria_dict and the model's output.")
            print("Consider updating your criteria_dict or adjusting your model prompt.")

        try:
            df, struct_missing_keys = self.structure_output(answerset=record2answer)
        except Exception as e:
            self.logger.error(f"Error in structure_output for record2answer: {record2answer}. Exception: {e}")
            raise

        df = self.select_articles_based_on_criteria(df)

        if struct_missing_keys:
            print(f"Warning: The following criteria were missing in some records: {', '.join(struct_missing_keys)}")
            print("This may indicate a mismatch between your criteria_dict and the model's output.")
            print("Consider updating your criteria_dict or adjusting your model prompt.")

        results = self.merge_results(df, screening_type)
        setattr(self, results_attr, results)
        # print(f"Columns in {results_attr}:", results.columns)

        return results  # Return the results for further processing if needed

    def merge_results(self, df, screening_type):
        if screening_type == 'screening1':
            return self.literature_df.merge(df, on='uniqueid', how='right')
        elif screening_type == 'screening2':
            return self.results_screening2.merge(df, on='uniqueid', how='right')
        else:
            raise ValueError(f"Invalid screening type: {screening_type}")

    def analyze_model_output(self, screening_type):
        all_keys = set()
        inconsistent_records = []
        record2answer = getattr(self, f"{screening_type}_record2answer")

        for record, answers in record2answer.items():
            if isinstance(answers, str):
                try:
                    # Try to parse the string as JSON
                    answers_dict = json.loads(answers)
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse answer for record {record} as JSON. Skipping this record.")
                    inconsistent_records.append(record)
                    continue
            elif isinstance(answers, dict):
                answers_dict = answers
            else:
                self.logger.warning(
                    f"Unexpected type for answers of record {record}: {type(answers)}. Skipping this record.")
                inconsistent_records.append(record)
                continue

            record_keys = set(answers_dict.keys())
            all_keys.update(record_keys)

            if record_keys != set(self.criteria_dict.keys()):
                inconsistent_records.append(record)

        extra_keys = all_keys - set(self.criteria_dict.keys())
        missing_keys = set(self.criteria_dict.keys()) - all_keys

        if extra_keys:
            self.logger.warning(f"Extra keys found in model output: {extra_keys}")
        if missing_keys:
            self.logger.warning(f"Keys missing from model output: {missing_keys}")

        return inconsistent_records, extra_keys, missing_keys

    def download_documents(self, df: pd.DataFrame = None, outdir: str = None) -> pd.DataFrame:
        """
        Modified to use ArticleDownloader class
        """
        if outdir is not None:
            self.pdf_location = outdir
        if df is not None:
            self.results_screening2 = df.copy(deep=True)

        # Use ArticleDownloader to process the DataFrame
        self.results_screening2 = self.article_downloader.process_dataframe(
            self.results_screening2,
            checkpoint_file=f"{self.vector_store_path}/download_progress.pkl"
        )

        # Update screening2_PDFdownload_error from ArticleDownloader
        self.screening2_PDFdownload_error = self.article_downloader.pdf_download_error

        # Save the updated error set
        with open(f'{self.embeddings_path2}/screening2_PDFdownload_error.pkl', 'wb') as file:
            pickle.dump(self.screening2_PDFdownload_error, file)

        self.logger.info(
            f"Total PDFs available: {len(self.results_screening2[self.results_screening2['pdf_path'].notnull()])}\n"
            f"Total download/access errors: {len(self.screening2_PDFdownload_error)}"
        )

        return self.results_screening2

    # OBTAIN PDF DATA
    def download_and_embed_pdfs(self):
        # First, download the documents
        self.download_documents()

        # Then, embed the PDFs
        asyncio.run(self.embed_articles_PDFs())

    # PDF embeddings

    async def ensure_directory_permissions(self, directory):
        """
        Ensures the directory exists and has proper permissions.
        """
        try:
            os.makedirs(directory, exist_ok=True)
            os.chmod(directory, 0o777)  # Full permissions
            # Also ensure parent directory has proper permissions
            parent_dir = os.path.dirname(directory)
            os.chmod(parent_dir, 0o777)
        except Exception as e:
            self.logger.warning(f"Could not set permissions for {directory}: {e}")

    async def validate_chroma_db(self, path_name, collection_name):
        """
        Validates if a Chroma database is accessible and properly structured.
        Returns True if valid, False otherwise.
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

    async def embed_article_PDF(self, pdf_record: pd.Series = None):
        """
        Embeds a single PDF article with improved error handling and database management.
        pdf_record = record
        """
        uniqueid = str(pdf_record['uniqueid'])
        try:
            self.logger.debug(f"Starting embedding for article {uniqueid}")

            # Validate PDF path
            if pd.isna(pdf_record['pdf_path']) or not os.path.exists(pdf_record['pdf_path']):
                raise FileNotFoundError(f"PDF file not found: {pdf_record['pdf_path']}")

            # Define the text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )

            # Load and split the PDF with proper error handling
            async def load_and_split_pdf():
                try:
                    loader = PDFPlumberLoader(pdf_record['pdf_path'])
                    documents = await asyncio.to_thread(
                        lambda: loader.load_and_split(text_splitter=text_splitter)
                    )
                    if not documents:
                        raise ValueError("No text extracted from PDF")
                    return documents
                except Exception as e:
                    raise RuntimeError(f"PDF processing failed: {str(e)}")

            documents = await load_and_split_pdf()

            # Update metadata
            metadata = {k: str(v) for k, v in pdf_record.to_dict().items()}
            for doc in documents:
                doc.metadata.update(metadata)

            # Set up the Chroma database path
            path_name = f"{self.embeddings_path2}/pdf_{uniqueid}"
            await self.ensure_directory_permissions(path_name)

            # Validate existing database or create new one
            is_valid_db = await self.validate_chroma_db(path_name, f"pdf_{uniqueid}")

            if is_valid_db:
                self.logger.debug(f"Using existing Chroma index for article {uniqueid}")
                # Remove from error set upon success
                self.screening2_PDFembedding_error.discard(uniqueid)
                return Chroma(
                    collection_name=f"pdf_{uniqueid}",
                    persist_directory=path_name,
                    embedding_function=self.embeddings
                )

            # Create new database with retries
            max_retries = 3
            retry_delay = 1

            for attempt in range(max_retries):
                try:
                    pdf_db = Chroma(
                        collection_name=f"pdf_{uniqueid}",
                        persist_directory=path_name,
                        embedding_function=self.embeddings
                    )

                    # Process documents in batches
                    batch_size = 50
                    for i in range(0, len(documents), batch_size):
                        batch = documents[i:i + batch_size]
                        await pdf_db.aadd_documents(batch)
                        await asyncio.sleep(0.1)  # Prevent database locking

                    self.logger.debug(f"Successfully embedded article {uniqueid}")
                    # Remove from error set upon successful embedding
                    self.screening2_PDFembedding_error.discard(uniqueid)
                    return pdf_db

                except sqlite3.OperationalError as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Database error for {uniqueid}, attempt {attempt + 1}/{max_retries}")
                        await asyncio.sleep(retry_delay)
                        continue
                    raise
                except Exception as e:
                    raise RuntimeError(f"Database operation failed: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error embedding {uniqueid}: {e}")
            self.screening2_PDFembedding_error.add(uniqueid)
            return None

    async def get_embedded_pdfs(self):
        """
        Returns a set of uniqueids for successfully embedded PDFs with validation,
        but only for the uniqueids needed in screening 2.
        """
        embedded_pdfs = set()

        try:
            base_path = self.embeddings_path2
            if not os.path.exists(base_path):
                self.logger.warning(f"Embeddings directory does not exist: {base_path}")
                return embedded_pdfs

            # Check only for the uniqueids needed in screening 2
            for uniqueid in self.results_screening2['uniqueid'].astype(str):
                path_name = os.path.join(base_path, f"pdf_{uniqueid}")
                # self.logger.debug(f"Checking uniqueid: {uniqueid}, Path: {path_name}")  # Add this

                # Validate the database
                if await self.validate_chroma_db(path_name, f"pdf_{uniqueid}"):
                    embedded_pdfs.add(uniqueid)
                else:
                    self.logger.debug(f"Invalid or missing database for PDF {uniqueid}")   # Changed to debug level

        except Exception as e:
            self.logger.error(f"Error scanning embedded PDFs: {str(e)}")

        return embedded_pdfs

    async def embed_articles_PDFs(self):
        """
        Manages the PDF embedding process with improved error handling and progress tracking.
        """
        try:
            self.logger.info("Starting PDF embedding process...")

            # Ensure base directory exists with proper permissions
            await self.ensure_directory_permissions(self.embeddings_path2)

            # Configure error logging
            error_log_path = os.path.join(self.embeddings_path2, 'pdf_embedding_errors.log')
            logging.basicConfig(
                filename=error_log_path,
                level=logging.ERROR,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )

            # Load or initialize error tracking
            error_tracking_path = f'{self.embeddings_path2}/screening2_PDFembedding_error.pkl'
            if os.path.exists(error_tracking_path):
                with open(error_tracking_path, 'rb') as file:
                    self.screening2_PDFembedding_error = pickle.load(file)
            else:
                self.screening2_PDFembedding_error = set()

            # --- Debugging: Check for duplicates in results_screening2 ---
            num_duplicates = self.results_screening2['uniqueid'].duplicated().sum()
            if num_duplicates > 0:
                self.logger.warning(f"Found {num_duplicates} duplicate unique IDs in results_screening2")
            # ---

            # Get already embedded PDFs (only for the relevant uniqueids)
            # Get already embedded PDFs (vector DB entries) for the relevant unique IDs
            embedded_pdfs = await self.get_embedded_pdfs()
            self.logger.info(f"Found {len(embedded_pdfs)} already embedded PDFs")

            # --- Count uniqueids before filtering ---
            total_uniqueids_in_screening2 = len(self.results_screening2['uniqueid'].unique())
            self.logger.info(f"Total unique IDs in screening2 results: {total_uniqueids_in_screening2}")
            self.logger.info(
                f"Found {len(embedded_pdfs)} already embedded PDFs (out of {total_uniqueids_in_screening2} in screening2)"
            )
            # ---

            # Filter records needing processing
            records_to_process = []
            for _, record in self.results_screening2.iterrows():
                uniqueid = str(record['uniqueid'])

                # FIRST: Check if a vector DB entry already exists
                if uniqueid in embedded_pdfs:
                    self.logger.debug(f"Skipping uniqueid {uniqueid} because it already has a vector DB entry.")
                    continue

                # SECOND: Check for a valid PDF path
                if pd.isna(record['pdf_path']) or not os.path.exists(record['pdf_path']):
                    self.screening2_PDFembedding_error.add(uniqueid)
                    self.logger.debug(f"Adding uniqueid {uniqueid} to error set because of invalid PDF path.")
                    continue

                # THIRD (optional): If you want to retry those that were marked as errors before,
                # you could log that here—but since they aren’t in embedded_pdfs, they can be reprocessed.
                if uniqueid in self.screening2_PDFembedding_error:
                    self.logger.debug(f"Reprocessing uniqueid {uniqueid} from error set.")

                records_to_process.append(record)

            if not records_to_process:
                self.logger.info("No new PDFs to embed")
                return

            self.logger.info(f"Number of unique IDs to be processed: {len(records_to_process)}")

            # Process PDFs with controlled concurrency
            sem = asyncio.Semaphore(5)

            async def process_with_semaphore(record):
                async with sem:
                    return await self.embed_article_PDF(record)

            tasks = [
                asyncio.create_task(process_with_semaphore(record))
                for record in records_to_process
            ]

            # Track progress
            success_count = error_count = 0
            with tqdm(total=len(tasks), desc="Embedding PDFs") as progress_bar:
                for task in asyncio.as_completed(tasks):
                    try:
                        pdf_db = await task
                        if pdf_db is not None:
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
            with open(error_tracking_path, 'wb') as file:
                pickle.dump(self.screening2_PDFembedding_error, file)

            self.logger.info(
                f"PDF embedding completed:\n"
                f"Successfully embedded: {success_count}\n"
                f"Failed to embed: {error_count}\n"
                f"Total PDFs processed: {len(tasks)}"
            )

        except Exception as e:
            self.logger.error(f"Critical error in PDF embedding process: {str(e)}")
            raise

    # GENERATE REPORTS

    def merge_screening1_embeddings(self, literature_df: pd.DataFrame = None):
        """
        Takes the embeddings of the literature DataFrame and merges the resulting vectorDBs in chunks.
        Args:
            literature_df:

        Returns:
            The merged vectorDB.
        """
        if literature_df is not None:
            self.literature_df = literature_df

        # Load the embeddings
        self.vectorDB_screening1 = self.embed_literature_df(path_name=self.embeddings_path1)
        fig = self.visualizer.plot_vector_DB(self.vectorDB_screening1)
        return fig

    def merge_screening2_vector_dbs(self):
        """
        Merges all individual PDF Chroma vector DBs (for screening2) that passed screening2
        into a single merged Chroma index. This function skips any article that is already
        incorporated into the merged DB so that the code is safe to run multiple times.

        Returns:
            merged_db (Chroma): A merged Chroma index containing all new documents from the individual PDF DBs.
        """
        import tqdm

        # Directory for the merged DB
        merged_db_path = os.path.join(self.embeddings_path2, "merged_screening2")
        os.makedirs(merged_db_path, exist_ok=True)

        # Initialize the merged Chroma index
        merged_db = Chroma(
            collection_name="screening2_merged",
            persist_directory=merged_db_path,
            embedding_function=self.embeddings
        )

        # Retrieve already merged unique IDs from the merged DB.
        existing = merged_db._collection.get(include=["metadatas"])
        existing_uniqueids = set()
        for metadata in existing.get("metadatas", []):
            if metadata and "uniqueid" in metadata:
                existing_uniqueids.add(metadata["uniqueid"])

        self.logger.info(f"{len(existing_uniqueids)} article(s) already merged.")

        # Get all screening2 unique IDs (as strings)
        screening_uniqueids = self.results_screening2['uniqueid'].astype(str).unique()

        # Process with a progress bar
        new_count = 0
        for uniqueid in tqdm.tqdm(screening_uniqueids, desc="Merging screening2 DBs"):
            if uniqueid in existing_uniqueids:
                continue

            pdf_dir = os.path.join(self.embeddings_path2, f"pdf_{uniqueid}")
            if not os.path.exists(pdf_dir):
                self.logger.warning(f"Directory for PDF {uniqueid} does not exist. Skipping.")
                continue

            # Validate the individual PDF vector DB (calling the async validator in sync context)
            valid = asyncio.run(self.validate_chroma_db(pdf_dir, f"pdf_{uniqueid}"))
            if not valid:
                self.logger.warning(f"Chroma DB for PDF {uniqueid} is not valid. Skipping.")
                continue

            # Load the individual PDF vector DB
            pdf_db = Chroma(
                collection_name=f"pdf_{uniqueid}",
                persist_directory=pdf_dir,
                embedding_function=self.embeddings
            )

            # Retrieve stored documents and metadata
            results = pdf_db._collection.get(include=["documents", "metadatas"])
            docs = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            if docs and metadatas:
                merged_db.add_texts(docs, metadatas=metadatas)
                new_count += 1
                self.logger.info(f"Merged {len(docs)} doc(s) from PDF {uniqueid}.")
            else:
                self.logger.warning(f"No documents found in PDF {uniqueid}.")

        # Remove the persist() call since it is not available.
        self.logger.info(f"Finished merging. Added {new_count} new article(s).")
        self.logger.info(f"Merged screening2 vector DB is located at: {merged_db_path}")
        return merged_db

    def create_PRISMA_visualization(self, save_results=True):
        """
        Visualizes the PRISMA flow diagram based on the screening results.
        Returns:
            Plotly figure.
        """
        df = self.literature_df.copy()
        # Map values for predicted_screening1 and predicted_screening2
        df['analyzed_screening1'] = ~df['uniqueid'].isin(self.screening1_missing_records)
        df['predicted_screening1'] = df['uniqueid'].map(
            dict(zip(self.results_screening1['uniqueid'], self.results_screening1['predicted_screening1'])))
        df['PDF_downloaded'] = (
                (df['predicted_screening1'] == True) & (~df['uniqueid'].isin(self.screening2_PDFdownload_error)))
        df['PDF_embedding_error'] = (
                (df['predicted_screening1'] == True) & (~df['uniqueid'].isin(self.screening2_PDFdownload_error)) & (
            df['uniqueid'].isin(self.screening2_PDFembedding_error)))
        df['predicted_screening2'] = df['uniqueid'].map(
            dict(zip(self.results_screening2['uniqueid'], self.results_screening2['predicted_screening2'])))

        prisma_fig = self.visualizer.visualize_PRISMA(df)
        if save_results:
            # prisma_fig.write_image(f"{self.vector_store_path}/prisma_flow_diagram.png")
            html_path = os.path.join(self.vector_store_path, "prisma_flow_diagram.html")
            prisma_fig.write_html(html_path)
            self.logger.info(f"PRISMA flow diagram saved as {html_path}")
            print(f"PRISMA flow diagram saved as {html_path}")

        return prisma_fig

    def generate_excel_report(self, screening_type='screening1'):
        """
        Generates an Excel report of the screening results, including the label (True/False) and reason
        for each criterion for each article, with color coding.

        Args:
            screening_type (str): 'screening1' or 'screening2' to specify which screening results to use.
        """
        import xlsxwriter

        filename = f"{self.vector_store_path}/{screening_type}_screening_report.xlsx"

        # Select the appropriate record2answer dictionary and DataFrame
        if screening_type == 'screening1':
            record2answer = self.screening1_record2answer
            if self.results_screening1 is not None:
                article_df = self.results_screening1.copy()
            else:
                article_df = self.literature_df.copy()
        elif screening_type == 'screening2':
            record2answer = self.screening2_record2answer
            if self.results_screening2 is not None:
                article_df = self.results_screening2.copy()
            else:
                raise ValueError("No screening2 results found. Please run screening2 first.")
        else:
            raise ValueError("screening_type must be 'screening1' or 'screening2'")

        # Prepare a list to hold the data
        data = []

        # Loop over each article and extract the labels and reasons
        for uniqueid, answers in record2answer.items():
            row = {'uniqueid': uniqueid}
            for criterion in self.criteria_dict.keys():
                if criterion in answers:
                    # Directly access 'label' and 'reason' without .get()
                    label = answers[criterion]['label'] if isinstance(answers[criterion], dict) else answers[criterion]
                    reason = answers[criterion]['reason'] if isinstance(answers[criterion], dict) else None
                else:
                    label = None
                    reason = None

                # Ensure label is boolean
                if isinstance(label, bool):
                    pass
                elif isinstance(label, str):
                    label_lower = label.lower().strip()
                    if label_lower in ['true', 't', 'yes']:
                        label = True
                    elif label_lower in ['false', 'f', 'no']:
                        label = False
                    else:
                        label = False  # Default to False if not recognized
                else:
                    label = False  # Default to False if not recognized

                row[f'{criterion}_label'] = label
                row[f'{criterion}_reason'] = reason
            data.append(row)

        if not data:
            print(f"No valid records found for {screening_type}")
            return

        # Create a DataFrame from the data
        df_answers = pd.DataFrame(data)

        # Merge the answers DataFrame with the article DataFrame
        merged_df = pd.merge(article_df, df_answers, on='uniqueid', how='right')

        # Reorder columns to group logically: uniqueid, title/record, then alternating label and reason
        base_cols = ['uniqueid']
        if 'title' in merged_df.columns:
            base_cols.append('title')
        elif self.content_column in merged_df.columns:
            base_cols.append(self.content_column)

        # Get criteria columns without '_label' suffix for pairing
        criteria_cols = [col.replace('_label', '') for col in merged_df.columns
                         if col.endswith('_label') and col != 'predicted_screening_label']

        # Create pairs of label and reason columns
        paired_cols = []
        for criterion in criteria_cols:
            paired_cols.extend([f"{criterion}_label", f"{criterion}_reason"])

        # Add 'predicted_screening' at the end if it exists
        if 'predicted_screening' in merged_df.columns:
            paired_cols.append('predicted_screening')
        elif f'predicted_{screening_type}' in merged_df.columns: # Add this to account for different screening type names
            paired_cols.append(f'predicted_{screening_type}')

        # Combine all columns in desired order
        merged_df = merged_df[base_cols + paired_cols]

        # Now, write the merged_df to Excel with formatting
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            merged_df.to_excel(writer, sheet_name=screening_type, index=False)

            workbook = writer.book
            worksheet = writer.sheets[screening_type]

            # Define formats
            true_format = workbook.add_format({'bg_color': '#75E69D'})  # Light green
            false_format = workbook.add_format({'bg_color': '#E66F60'})  # Light red
            wrap_format = workbook.add_format({'text_wrap': True})

            # Set column widths
            worksheet.set_column('A:A', 10)  # uniqueid
            if 'title' in merged_df.columns:
                col_idx = merged_df.columns.get_loc('title')
                worksheet.set_column(col_idx, col_idx, 40)
            if 'Title' in merged_df.columns:
                col_idx = merged_df.columns.get_loc('Title')
                worksheet.set_column(col_idx, col_idx, 40)
            elif self.content_column in merged_df.columns:
                col_idx = merged_df.columns.get_loc(self.content_column)
                worksheet.set_column(col_idx, col_idx, 80, wrap_format)

            # Apply conditional formatting to boolean columns and set column widths
            for col in merged_df.columns:
                col_idx = merged_df.columns.get_loc(col)
                col_letter = xlsxwriter.utility.xl_col_to_name(col_idx)
                last_row = len(merged_df)
                if col.endswith('_label') or col == f'predicted_{screening_type}':
                    worksheet.set_column(col_idx, col_idx, 15)
                    worksheet.conditional_format(
                        f'{col_letter}2:{col_letter}{last_row + 1}',
                        {'type': 'cell',
                         'criteria': '==',
                         'value': True,
                         'format': true_format}
                    )
                    worksheet.conditional_format(
                        f'{col_letter}2:{col_letter}{last_row + 1}',
                        {'type': 'cell',
                         'criteria': '==',
                         'value': False,
                         'format': false_format}
                    )
                elif col.endswith('_reason'):
                    worksheet.set_column(col_idx, col_idx, 40, wrap_format)

            # Freeze panes to keep headers visible
            worksheet.freeze_panes(1, 0)

        print(f"Excel report saved to {filename}")

    def chatbot(self):
        """
        Launches the chatbot interface for interacting with the screening results.
        """

        # Assume academate_instance is your academate object, and it has an attribute `llm`
        chat_app = ChatbotApp(llm=self.llm)
        self.vectorDB_screening2 = self.merge_screening2_vector_dbs()

        retriever = self.vectorDB_screening2.as_retriever(
            search_type="mmr",
            search_kwargs={
                'score_threshold': 0.0,
                'fetch_k': 10,
                'k': 20
            }
        )
        prompt = "Costs and utilities of manual therapy and orthopedic standard care for low-prioritized orthopedic outpatients of working age: a cost consequence analysis"
        inputs = {
            "question": prompt,  # or the user question
            "retriever": retriever,
        }

        final_state = chat_app.invoke(inputs)
        return final_state

