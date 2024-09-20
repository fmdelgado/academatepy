import os.path
import pickle
import sys
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
import os
from tqdm import tqdm
import pandas as pd
import time
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from habanero import Crossref
from langchain.chains import RetrievalQA
import io
from langchain.prompts import ChatPromptTemplate
import json
# from app.research import Research
from app.dataset_visualizer import DatasetVisualizer
import logging
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import requests
from typing import Any, TypedDict
from paperscraper.pdf import save_pdf
import re

email = 'fernando.miguel.delgado-chaves@uni-hamburg.de'

"""
Defines data structures for storing document and research information.
"""


class DocumentInfo(TypedDict):
    keywords: str | None
    pmid: str | None
    title: str | None
    abstract: str | None
    journal: str | None
    doi: str | None
    authors: str | None
    publication_date: str | None
    record: str | None


class ResearchInfo(TypedDict):
    research_topic: str | None
    # criteria_dict: dict | None
    research_search_gs: int | None
    research_search_pubmed: int | None
    doc_with_missing_abstract: int | None
    doublicated_count: int | None
    final_search_count: int | None
    screening_1: int | None
    screening_2: int | None
    final_selection: int | None
    visualize_vectorDB: Any | None
    visualize_screening: Any | None
    visualize_PRISMA: Any | None


class academate:
    def __init__(self, topic: str, llm: object, embeddings: object, criteria_dict: dict, vector_store_path: str,
                 literature_df: pd.DataFrame = None, content_column: str = "record", embeddings_path: str = None,
                 use_pubmed: bool = True, use_googlescholar: bool = True, use_gs_proxy: bool = True,
                 verbose: bool = False, chunksize: int = 25) -> None:

        self.embeddings_path = embeddings_path
        self.visualizer = DatasetVisualizer(llm)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # Initial attributes
        self.topic: str = topic
        self.criteria_dict = criteria_dict
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store_path = vector_store_path
        self.literature_df = literature_df
        if self.embeddings_path is None:
            self.embeddings_path = f"{self.vector_store_path}/embeddings"
        if self.literature_df is not None:
            self.literature_df.reset_index(drop=True, inplace=True)
            self.literature_df['uniqueid'] = self.literature_df.index.astype(str)
        self.content_column = content_column
        self.verbose = verbose  # TO BE IMPLEMENTED

        # Screening attributes
        self.selected_fetch_k = 20
        self.selected_k = len(self.criteria_dict)

        # Screening 1 attributes
        self.results_screening1 = None
        self.screening1_record2answer = dict()
        self.screening1_missing_records = set()
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

        self.pdf_location = f"{self.vector_store_path}/pdfs"
        if not os.path.exists(self.pdf_location):
            os.makedirs(self.pdf_location)
            os.chmod(self.pdf_location, 0o777)

        # chatting
        self.db_qa = None
        self.analysis_time = None
        self.db = None

    # EMBEDDINGS
    def docs_to_FAISS(self, docs, path_name):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=400,
        )
        # docs = loader.load()
        splits = text_splitter.split_documents(docs)
        index = FAISS.from_documents(splits, self.embeddings)
        index.save_local(path_name)
        return index

    def load_FAISS(self, path_name):
        index = FAISS.load_local(path_name, self.embeddings, allow_dangerous_deserialization=True)
        return index

    def embed_literature_df(self, path_name=None):
        if path_name is None:
            path_name = self.embeddings_path1
        if self.verbose:
            print("Number of records: ", len(self.literature_df))
        loader = DataFrameLoader(self.literature_df, page_content_column=self.content_column)
        # If embedding already present in the vector store, load it
        if os.path.exists(path_name):
            if self.verbose:
                print('loading FAISS DB')
            self.db = self.load_FAISS(path_name=path_name)
            num_documents = len(self.db.index_to_docstore_id)
            # print(f"Total number of documents: {num_documents}")
        else:
            t1 = time.time()
            if self.verbose:
                print('Creating new FAISS DB')
            self.db = self.docs_to_FAISS(loader.load(), path_name=path_name)
            if self.verbose:
                print("Time taken to create FAISS DB: ", time.time() - t1)
        # time.sleep(10)
        return self.db

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

        large_literature_df = self.literature_df.copy()
        vectordb_list = []
        for start in tqdm(range(0, len(large_literature_df), self.chunksize), desc="Merging screening 1 embeddings"):
            end = start + self.chunksize
            self.literature_df = large_literature_df[start:end].copy()
            subvector_db = self.embed_literature_df(path_name=f"{self.embeddings_path1}/lit_chunk_{start}_{end}")
            vectordb_list.append(subvector_db)
        self.literature_df = large_literature_df

        for i, db1 in enumerate(vectordb_list):
            if i == 0:
                self.vectorDB_screening1 = db1
            else:
                self.vectorDB_screening1.merge_from(db1)

        self.vectorDB_screening1.save_local(f"{self.screening1_dir}/vectorDB_screening1")
        fig = self.visualizer.plot_vector_DB(self.vectorDB_screening1)
        return fig

    # SCREENING

    def parse_json_safely(self, json_string):
        try:
            # Extract the JSON object from the response using regex
            json_matches = re.findall(r'\{.*\}', json_string, re.DOTALL)
            if json_matches:
                json_data = json_matches[0]
                return json.loads(json_data)
            else:
                self.logger.error(f"No JSON found in the response: {json_string}")
                return {}  # Return an empty dict if parsing fails
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}\nResponse was: {json_string}")
            return {}

    def prepare_screening_prompt(self):
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
        and a brief reason for your decision.

        **Important**: Respond **only** with a JSON object matching the following structure, and do not include any additional text:

        {json_structure}

        Ensure your response is a valid JSON object.
        """)

        return prompt, formatted_criteria, json_structure

    def prepare_chain(self, retriever, formatted_criteria, json_structure, prompt):
        rag_chain_from_docs = (
                RunnableParallel(
                    {
                        "context": lambda x: "\n\n".join(
                            [doc.page_content for doc in retriever.invoke(x)]),
                        "criteria": lambda x: formatted_criteria,
                        "json_structure": lambda x: json_structure
                    }
                )
                | prompt
                | self.llm
                | (lambda x: self.parse_json_safely(x.content))
        )
        chain = RunnablePassthrough() | rag_chain_from_docs
        return chain

    def base_screening(self, screening_type):
        # screening_type = 'screening1'
        screening_dir = getattr(self, f"{screening_type}_dir")
        record2answer_attr = f"{screening_type}_record2answer"
        missing_records_attr = f"{screening_type}_missing_records"

        # Load existing screening data
        self.load_existing_screening_data(screening_type)
        record2answer = getattr(self, record2answer_attr)
        missing_records = getattr(self, missing_records_attr)

        prompt, formatted_criteria, json_structure = self.prepare_screening_prompt()

        selected_k = len(self.criteria_dict)
        selected_fetch_k = 20
        rec_numbers = self.get_rec_numbers(screening_type)

        init_time = time.time()

        # Create a new list of records to process, excluding those already in record2answer
        records_to_process = [rec for rec in rec_numbers if rec not in record2answer.keys()]
        # recnumber = 0
        for i, recnumber in enumerate(tqdm(records_to_process, desc=f"{screening_type.capitalize()}")):
            try:
                retriever = self.get_retriever(recnumber, screening_type)
                if retriever is None:
                    raise ValueError(f"Failed to create retriever for record {recnumber}")
                chain = self.prepare_chain(retriever, formatted_criteria, json_structure, prompt)
                result = self.process_single_record(recnumber, chain)

                if result is None:
                    missing_records.add(recnumber)
                    self.logger.warning(f"Failed to process record {recnumber} after multiple attempts.")
                else:
                    record2answer[recnumber] = result
                    missing_records.discard(recnumber)  # Remove from missing if it was there

                if i % 20 == 0:
                    self.save_screening_results(screening_type)
            except Exception as e:
                self.logger.error(f"Error processing record {recnumber}: {str(e)}")
                missing_records.add(recnumber)

        # Final check to ensure all records are accounted for
        all_records = set(rec_numbers)
        analyzed_records = set(record2answer.keys())
        missing_records = all_records - analyzed_records

        self.save_screening_results(screening_type)
        setattr(self, f"{screening_type}_analysis_time", time.time() - init_time)

        if self.verbose:
            print(f"Total records: {len(all_records)}")
            print(f"Successfully analyzed: {len(analyzed_records)}")
            print(f"Missing/Failed: {len(missing_records)}")

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

        if os.path.exists(missing_records_path):
            with open(missing_records_path, 'rb') as file:
                setattr(self, missing_records_attr, pickle.load(file))
        else:
            setattr(self, missing_records_attr, set())

        if self.verbose:
            print(f"Loaded {len(getattr(self, record2answer_attr))} existing records for {screening_type}")
            print(f"Loaded {len(getattr(self, missing_records_attr))} missing records for {screening_type}")

    def get_rec_numbers(self, screening_type):
        if screening_type == 'screening1':
            return self.literature_df['uniqueid'].astype(str).to_list()
        else:  # screening2
            return self.results_screening2['uniqueid'].astype(str).to_list()

    def get_retriever(self, recnumber, screening_type):
        if screening_type == 'screening1':
            return self.db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    'fetch_k': self.selected_fetch_k,
                    'k': self.selected_k,
                    'filter': {'uniqueid': recnumber}
                }
            )
        else:  # screening2
            pdf_record = self.results_screening2[self.results_screening2['uniqueid'] == recnumber].iloc[0]
            self.embed_article_PDF(pdf_record)
            return self.db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    'fetch_k': self.selected_fetch_k,
                    'k': self.selected_k,
                    'filter': {'uniqueid': recnumber}
                }
            )

    def process_single_record(self, recnumber, chain):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Invoke the chain with the recnumber as a string
                result = chain.invoke(f"Analyze document {recnumber}")

                if result:
                    return result
                else:
                    self.logger.warning(f"Unexpected response format for record {recnumber}. Retrying...")
            except Exception as e:
                self.logger.error(f"Error processing record {recnumber}: {str(e)}")

            if attempt < max_retries - 1:
                self.logger.info(f"Retrying record {recnumber}... (Attempt {attempt + 2}/{max_retries})")
            else:
                self.logger.error(f"Failed to process record {recnumber} after {max_retries} attempts.")

        return None

    def save_screening_results(self, screening_type):
        screening_dir = getattr(self, f"{screening_type}_dir")
        record2answer = getattr(self, f"{screening_type}_record2answer")
        missing_records = getattr(self, f"{screening_type}_missing_records")

        with open(f'{screening_dir}/{screening_type}_predicted_criteria.pkl', 'wb') as file:
            pickle.dump(record2answer, file)
        with open(f'{screening_dir}/{screening_type}_missing_records.pkl', 'wb') as file:
            pickle.dump(missing_records, file)

    def structure_output(self, answerset: dict = None):
        data_dict = {}
        missing_keys = set()
        for key in answerset.keys():
            data_dict[key] = {}
            for checkpoint_studied in self.criteria_dict.keys():
                if checkpoint_studied in answerset[key]:
                    data_dict[key][checkpoint_studied] = answerset[key][checkpoint_studied].get('label', False)
                else:
                    data_dict[key][checkpoint_studied] = False
                    missing_keys.add(checkpoint_studied)

        if missing_keys and self.verbose:
            print(f"Warning: The following keys were not found in some records: {', '.join(missing_keys)}")

        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df['uniqueid'] = df.index.astype(str)
        df.reset_index(inplace=True, drop=True)
        return df, missing_keys

    def select_articles_based_on_criteria(self, df: pd.DataFrame = None):
        if self.verbose:
            print("Selecting articles based on criteria...")
        for column in self.criteria_dict.keys():
            df[column] = df[column].astype(bool)

        columns_needed_as_true = list(self.criteria_dict.keys())
        df['predicted_screening'] = df[columns_needed_as_true].all(axis=1)
        return df

    def run_screening1(self):
        if self.verbose:
            print("Total number of records: ", len(self.literature_df))
        large_literature_df = self.literature_df.copy()

        start_time = time.time()
        for start in tqdm(range(0, len(large_literature_df), self.chunksize), desc="Screening 1"):
            end = start + self.chunksize
            self.literature_df = large_literature_df[start:end].copy()
            db = self.embed_literature_df(path_name=f"{self.embeddings_path1}/lit_chunk_{start}_{end}")
            self.base_screening('screening1')
        self.literature_df = large_literature_df.copy()

        self.analysis_time = time.time() - start_time
        self.results_screening1 = self.post_screening_analysis('screening1')
        self.results_screening1['runtime_scr1'] = self.analysis_time
        # rename criteria columns to add '_scr1'
        self.results_screening1.rename(columns={key: key + '_scr1' for key in self.criteria_dict.keys()}, inplace=True)

        # Add this line to rename 'predicted_screening' to 'predicted_screening1'
        self.results_screening1.rename(columns={'predicted_screening': 'predicted_screening1'}, inplace=True)

        if self.verbose:
            print("Columns in results_screening1:", self.results_screening1.columns)

        return self.results_screening1

    def run_screening2(self):
        if self.results_screening1 is None:
            if 'pdf_path' not in self.literature_df.columns:
                raise ValueError(
                    "Either run_screening1() must be called before run_screening2() or 'pdf_path' column must be provided in the initial DataFrame.")
            else:
                print("Using provided PDF paths for screening2.")
                self.results_screening2 = self.literature_df.copy()
        else:
            if 'predicted_screening1' not in self.results_screening1.columns:
                print("Warning: 'predicted_screening1' column not found. Using all records for screening2.")
                self.results_screening2 = self.results_screening1.copy()
            else:
                self.results_screening2 = self.results_screening1[
                    self.results_screening1['predicted_screening1'] == True]

        if self.verbose:
            print("Records selected for PDF processing: ", len(self.results_screening2))

        # If PDFs are not already downloaded, download them
        if 'pdf_path' not in self.results_screening2.columns or self.results_screening2['pdf_path'].isnull().any():
            self.download_and_embed_pdfs()
        else:
            self.embed_articles_PDFs()

        # Filter out records where PDF is not available
        self.results_screening2 = self.results_screening2[self.results_screening2['pdf_path'].notnull()]

        if self.verbose:
            print("Records with PDF proceeding for screening2: ", len(self.results_screening2))

        if len(self.results_screening2) > 0:
            start_time = time.time()
            self.base_screening('screening2')
            self.analysis_time = time.time() - start_time
            self.results_screening2 = self.post_screening_analysis('screening2')
            self.results_screening2['runtime_scr2'] = self.analysis_time
            # rename criteria columns to add '_scr2'
            self.results_screening2.rename(columns={key: key + '_scr2' for key in self.criteria_dict.keys()},
                                           inplace=True)

            # Add this line to rename 'predicted_screening' to 'predicted_screening2'
            self.results_screening2.rename(columns={'predicted_screening': 'predicted_screening2'}, inplace=True)

            if self.verbose:
                print("Columns in results_screening2:", self.results_screening2.columns)

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

        if self.verbose:
            print(f"Total records for {screening_type}: {total_records}")
            print(f"CORRECTLY analyzed: {correctly_analyzed}")
            print(f"INCORRECTLY analyzed: {incorrectly_analyzed}")

        inconsistent_records, extra_keys, missing_keys = self.analyze_model_output(screening_type)

        if inconsistent_records or extra_keys or missing_keys:
            print("There are inconsistencies between your criteria_dict and the model's output.")
            print("Consider updating your criteria_dict or adjusting your model prompt.")

        df, struct_missing_keys = self.structure_output(answerset=record2answer)
        df = self.select_articles_based_on_criteria(df)

        if struct_missing_keys:
            print(f"Warning: The following criteria were missing in some records: {', '.join(struct_missing_keys)}")
            print("This may indicate a mismatch between your criteria_dict and the model's output.")
            print("Consider updating your criteria_dict or adjusting your model prompt.")

        results = self.merge_results(df, screening_type)
        setattr(self, results_attr, results)
        print(f"Columns in {results_attr}:", results.columns)

        return results  # Return the results for further processing if needed

    def merge_results(self, df, screening_type):
        if screening_type == 'screening1':
            return self.literature_df.merge(df, on='uniqueid', how='right')
        else:  # screening2
            return self.results_screening2.merge(df, on='uniqueid', how='right')

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

    # OBTAIN PDF DATA
    def download_and_embed_pdfs(self):
        # First, download the documents
        self.download_documents()

        # Then, embed the PDFs
        self.embed_articles_PDFs()

    def find_record_doi(self, df: pd.DataFrame = None):
        if df is not None:
            working_df = df.copy(deep=True)
        else:
            working_df = self.results_screening2.copy(deep=True)

        # Ensure 'doi' and 'open_access' columns exist
        if 'doi' not in working_df.columns:
            working_df['doi'] = None
        if 'open_access' not in working_df.columns:
            working_df['open_access'] = None

        # Define the path for saving/loading DOI information
        doi_cache_path = f"{self.vector_store_path}/doi_cache.pkl"

        # Check if we have cached DOI information
        if os.path.exists(doi_cache_path):
            with open(doi_cache_path, 'rb') as f:
                cached_doi_info = pickle.load(f)

            # Update the working_df with cached information
            for column in ['doi', 'open_access']:
                if column in cached_doi_info:
                    working_df[column] = working_df['uniqueid'].map(cached_doi_info[column])

        # Identify records that still need DOI retrieval
        records_to_process = working_df[working_df['doi'].isna()]['uniqueid'].tolist()

        if records_to_process:
            # Initialize logging
            logging.basicConfig(level=logging.INFO, filename='doi_retrieval.log', filemode='w',
                                format='%(asctime)s - %(levelname)s - %(message)s')

            # Initialize CrossRef API client
            cr = Crossref()
            email = 'your_email@example.com'  # Replace with your actual email address

            # Function to retrieve DOI
            def get_doi(row):
                query_parts = []
                for col in df_columns:
                    if col in row:
                        query_parts.append(str(row[col]))
                query = ' '.join(query_parts)

                attempt = 0
                while attempt < 3:
                    try:
                        result = cr.works(query=query)
                        items = result['message']['items']
                        if items:
                            return items[0].get('DOI', 'DOI not found')
                    except Exception as e:
                        logging.error(f"Attempt {attempt + 1} - An error occurred: {e}")
                        attempt += 1
                return 'DOI not found'

            # Define the function to check if the DOI is freely available
            def check_doi_freely_available(doi, email):
                try:
                    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
                    response = requests.get(url)
                    data = response.json()
                    if 'is_oa' in data:
                        return data['is_oa']
                except Exception as e:
                    logging.error(f"An error occurred while checking DOI {doi}: {e}")
                return False

            df_columns = set(working_df.drop(self.criteria_dict.keys(), errors='ignore', axis=1).columns)
            df_columns = df_columns - {self.content_column}

            # Process only the records that need DOI retrieval
            df_to_process = working_df[working_df['uniqueid'].isin(records_to_process)]

            # Apply the get_doi function with progress tracking
            tqdm.pandas(desc="Retrieving DOIs")
            df_to_process['doi'] = df_to_process.progress_apply(lambda row: get_doi(row), axis=1)

            # Apply the check_doi_freely_available function with progress tracking
            tqdm.pandas(desc="Checking DOI availability")
            df_to_process['open_access'] = df_to_process['doi'].progress_apply(
                lambda doi: check_doi_freely_available(doi, email=email))

            # Update the working_df with new information
            working_df.update(df_to_process)

            # Update the cached information
            cached_doi_info = {'doi': {}, 'open_access': {}}
            for idx, row in working_df.iterrows():
                cached_doi_info['doi'][row['uniqueid']] = row['doi']
                cached_doi_info['open_access'][row['uniqueid']] = row['open_access']

            # Save the updated cache
            with open(doi_cache_path, 'wb') as f:
                pickle.dump(cached_doi_info, f)

        self.results_screening2 = working_df
        return self.results_screening2

    def download_documents(self, df: pd.DataFrame = None, outdir: str = None) -> pd.DataFrame:
        if outdir is not None:
            self.pdf_location = outdir
        if df is not None:
            self.results_screening2 = df.copy(deep=True)

        self.find_record_doi(df=self.results_screening2)

        if not os.path.exists(self.pdf_location):
            os.makedirs(self.pdf_location, exist_ok=True)

        # Check if 'pdf_path' column already exists
        pdf_path_exists = 'pdf_path' in self.results_screening2.columns

        # Initialize or load the screening2_PDFdownload_error set
        if not hasattr(self, 'screening2_PDFdownload_error'):
            self.screening2_PDFdownload_error = set()

        for index, row in tqdm(self.results_screening2.iterrows(), total=self.results_screening2.shape[0],
                               desc="Processing articles"):
            if pdf_path_exists and pd.notna(row['pdf_path']):
                # If pdf_path is provided and not null, verify the file exists
                if os.path.isfile(row['pdf_path']):
                    self.screening2_PDFdownload_error.discard(row['uniqueid'])
                    continue
                else:
                    # If the file doesn't exist, we'll try to download it
                    self.results_screening2.at[index, 'pdf_path'] = None

            doi = row['doi']
            if pd.isna(doi) or doi == 'DOI not found':
                self.screening2_PDFdownload_error.add(row['uniqueid'])
                continue  # Skip this iteration if DOI is not available

            paper_data = {'doi': doi}
            name = doi.replace('/', '_').replace(':', '__').replace('.', '-')
            name += '.pdf'
            filepath = os.path.join(self.pdf_location, name)

            if not os.path.exists(filepath):
                try:
                    save_pdf(paper_data, filepath=filepath)
                    if os.path.isfile(filepath):
                        self.results_screening2.at[index, 'pdf_path'] = filepath
                        self.screening2_PDFdownload_error.discard(
                            row['uniqueid'])  # Remove from error set if successful
                    else:
                        self.results_screening2.at[index, 'pdf_path'] = None
                        self.screening2_PDFdownload_error.add(row['uniqueid'])
                except Exception as e:
                    self.logger.error(f"Error downloading PDF for {doi}: {str(e)}")
                    self.results_screening2.at[index, 'pdf_path'] = None
                    self.screening2_PDFdownload_error.add(row['uniqueid'])
            else:
                self.results_screening2.at[index, 'pdf_path'] = filepath
                self.screening2_PDFdownload_error.discard(row['uniqueid'])  # Remove from error set if file exists
                if self.verbose:
                    print(f"File {filepath} already exists. Skipping download.")

        # Save the updated screening2_PDFdownload_error set
        with open(f'{self.embeddings_path2}/screening2_PDFdownload_error.pkl', 'wb') as file:
            pickle.dump(self.screening2_PDFdownload_error, file)

        if self.verbose:
            print(
                f"Total PDFs available: {len(self.results_screening2[self.results_screening2['pdf_path'].notnull()])}")
            print(f"Total download/access errors: {len(self.screening2_PDFdownload_error)}")

        return self.results_screening2

    def embed_article_PDF(self, pdf_record: pd.Series = None):
        try:
            loader = PDFPlumberLoader(pdf_record['pdf_path'])
            pages = loader.load_and_split()

            # Add metadata of the article to the pages
            for page in pages:
                page.metadata.update(pdf_record.to_dict())

            path_name = f"{self.embeddings_path2}/pdf_{pdf_record['uniqueid']}"

            if os.path.exists(path_name):
                self.db = self.load_FAISS(path_name=path_name)
            else:
                t1 = time.time()
                if self.verbose:
                    print(f"Creating new FAISS DB for article {pdf_record['uniqueid']}")

                # Add error handling here
                embeddings = self.embeddings.embed_documents([page.page_content for page in pages])
                if not embeddings:
                    raise ValueError("Embedding process failed: no embeddings generated")

                self.db = self.docs_to_FAISS(pages, path_name=path_name)
                if self.verbose:
                    print("Time taken to create FAISS DB: ", time.time() - t1)
        except Exception as e:
            print(f"Error processing PDF {pdf_record['uniqueid']}: {str(e)}")
            self.screening2_PDFembedding_error.add(pdf_record['uniqueid'])
            return None
        return self.db

    def embed_articles_PDFs(self):
        if self.verbose:
            print("Embedding articles...")

            # Set up logging
        logging.basicConfig(filename='pdf_embedding_errors.log', level=logging.ERROR,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        if os.path.exists(f'{self.embeddings_path2}/screening2_PDFembedding_error.pkl'):
            with open(f'{self.embeddings_path2}/screening2_PDFembedding_error.pkl', 'rb') as file:
                self.screening2_PDFembedding_error = pickle.load(file)
        else:
            self.screening2_PDFembedding_error = set()

        for i, pdf_record in tqdm(self.results_screening2.iterrows(), desc="Embedding PDFs"):
            try:
                # Check if the file exists and is a valid PDF
                pdf_path = pdf_record.get('pdf_path')  # Adjust this to match your actual column name
                if not pdf_path or not os.path.exists(pdf_path):
                    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

                # Try to open the PDF file
                with open(pdf_path, 'rb') as file:
                    try:
                        PdfReader(file)
                    except PdfReadError as e:
                        raise ValueError(f"Invalid PDF file: {e}")

                # If we get here, the PDF is valid, so we can proceed with embedding
                self.embed_article_PDF(pdf_record)
                self.screening2_PDFembedding_error.discard(pdf_record['uniqueid'])

            except FileNotFoundError as e:
                logging.error(f"File not found for record {pdf_record['uniqueid']}: {str(e)}")
                self.screening2_PDFembedding_error.add(pdf_record['uniqueid'])

            except ValueError as e:
                logging.error(f"Invalid PDF for record {pdf_record['uniqueid']}: {str(e)}")
                self.screening2_PDFembedding_error.add(pdf_record['uniqueid'])

            except Exception as e:
                logging.error(f"Error processing record {pdf_record['uniqueid']}: {str(e)}")
                self.screening2_PDFembedding_error.add(pdf_record['uniqueid'])

            if self.verbose and pdf_record['uniqueid'] in self.screening2_PDFembedding_error:
                print(f"Error processing record {pdf_record['uniqueid']}")

        with open(f'{self.embeddings_path2}/screening2_PDFembedding_error.pkl', 'wb') as file:
            pickle.dump(self.screening2_PDFembedding_error, file)

    # GENERATE REPORTS
    def create_PRISMA_visualization(self):
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
            ~df['uniqueid'].isin(self.screening2_PDFembedding_error)))
        df['predicted_screening2'] = df['uniqueid'].map(
            dict(zip(self.results_screening2['uniqueid'], self.results_screening2['predicted_screening2'])))

        prisma_fig = self.visualizer.visualize_PRISMA(df)
        return prisma_fig

    def generate_excel_report(self, screening_type='screening1', filename='screening_report.xlsx'):
        """
        Generates an Excel report of the screening results, including the label (True/False) and reason
        for each criterion for each article.

        Args:
            screening_type (str): 'screening1' or 'screening2' to specify which screening results to use.
            filename (str): The filename of the Excel file to create.
        """
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
                    label = answers[criterion].get('label', None)
                    reason = answers[criterion].get('reason', None)
                else:
                    label = None
                    reason = None
                row[f'{criterion}_label'] = label
                row[f'{criterion}_reason'] = reason
            data.append(row)

        # Create a DataFrame from the data
        df_answers = pd.DataFrame(data)

        # Merge the answers DataFrame with the article DataFrame
        merged_df = pd.merge(article_df, df_answers, on='uniqueid', how='left')

        # Write the merged DataFrame to Excel
        merged_df.to_excel(filename, index=False)

        print(f"Excel report saved to {filename}")

    # CHATBOT
    def chat_with_articles(self):
        if self.vectorDB_screening2 is None:
            selected_articles = self.results_screening2[self.results_screening2['predicted_screening2'] == True]
            dbs_list = []
            for i, pdf_record in tqdm(selected_articles.iterrows(), desc="Gathering DBs"):
                # print(i, pdf_record['uniqueid'])
                db = self.embed_article_PDF(pdf_record)
                dbs_list.append(db)

            for i, db1 in enumerate(dbs_list):
                if i == 0:
                    self.vectorDB_screening2 = db1
                else:
                    self.vectorDB_screening2.merge_from(db1)

            self.vectorDB_screening2.save_local(f"{self.screening2_dir}/vectorDB_screening2")
        else:
            self.vectorDB_screening2 = self.load_FAISS(f"{self.screening2_dir}/vectorDB_screening2")

        retriever = self.vectorDB_screening2.as_retriever()

        PROMPT_TEMPLATE = """Answer the question based only on the following context:
        {context}
        You are allowed to rephrase the answer based on the context. 
        Question: {question}
        """
        PROMPT = PromptTemplate.from_template(PROMPT_TEMPLATE)

        # tag::qa[]
        self.db_qa = RetrievalQA.from_chain_type(
            self.llm,  # <1>
            chain_type="stuff",  # <2>
            retriever=retriever,  # <3>
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        return self.db_qa

    def generate_response(self, prompt):
        """
        Create a handler that calls the Conversational agent
        and returns a response to be rendered in the UI
        prompt = 'why is physiotherapy important?'
        """
        # Create a string buffer
        buffer = io.StringIO()
        # Redirect stdout to the buffer
        sys.stdout = buffer

        response = self.db_qa.invoke(prompt)

        # Reset stdout to its original value
        sys.stdout = sys.__stdout__
        # Get the verbose output from the buffer
        verbose_output = buffer.getvalue()
        return response
