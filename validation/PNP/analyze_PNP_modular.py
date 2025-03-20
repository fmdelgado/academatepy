import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import requests, json
from dotenv import load_dotenv
import os
import sys
# Import from the new academate package
sys.path.append("/home/bbb1417/academatepy/")
from academate import AcademateSingleDB as Academate  # This imports the standard implementation
import pickle
import numpy as np

# screen -S llm_reproduction_academate python reproduction_testing.py
test_result_type = "results_modular"

dotenv_path = '/home/bbb1417/academatepy/.env'
load_dotenv(dotenv_path)

# READING THE DF
workdir = "/home/bbb1417/academatepy/validation/PNP"
df = pd.read_pickle(f'/home/bbb1417/academatepy/validation/PNP/preprocessed_articles_filtered.pkl')
output_dir = "/home/bbb1417/academatepy/validation/PNP/results"

print(df.columns)

criteria_dict = {
    "Disease": "If the study investigates a condition that is classified as a form of hereditary peripheral neuropathy and is recognized based on its genetic basis in medical literature, return True. Otherwise, return False.",
    "Treatment": "If the study focuses on pharmacological therapies or genotype-related dietary changes specifically targeting the neuropathy, return True. If the study primarily investigates non-pharmacological interventions such as physiotherapy, surgery, and genetic counseling, return False.",
    "Human": "If the study involves human participants, return True. If it involves only animal models or in vitro studies, return False.",
    "Genetic": "If human participants in the study are confirmed to have hereditary peripheral neuropathy through direct genetic testing or through family genetic association where further individual testing might not be necessary, return True. This includes confirmation through nucleotide-based assays and clinically accepted diagnostic methods such as specific enzyme activity in biopsies, judged on a case-by-case basis to be as definitive as genetic tests. If the study includes participants diagnosed only based on clinical symptoms without genetic confirmation, return False.",
    "Results": "If results are reported specifically for the genetically confirmed cohort in studies including both genetically confirmed and non-confirmed participants, return True. If the study does not segregate data between genetically confirmed and non-confirmed participants, or if results from credible sources and databases are not exclusively from interventional studies with available results, return False."
}


### OLLAMA

# Read a pickle file
with open("/home/bbb1417/academatepy/validation/ollama_model_list.pkl", 'rb') as f:
    ollama_models = pickle.load(f)
# If you want to see what's in it
print(ollama_models)  # If it's a list/dict it will print directly

# Authentication details
protocol = "https"
hostname = "chat.cosy.bio"
host = f"{protocol}://{hostname}"
auth_url = f"{host}/api/v1/auths/signin"
api_url = f"{host}/ollama"
account = {
    'email': os.getenv("ollama_user"),
    'password': os.getenv("ollama_pw")
}
auth_response = requests.post(auth_url, json=account)
jwt = auth_response.json()["token"]
headers = {"Authorization": "Bearer " + jwt}

ollama_embeddings = OllamaEmbeddings(base_url=api_url, model='nomic-embed-text:latest',
                                     client_kwargs={'headers': headers})
#

for ollama_model in ollama_models:
    print("\n ANALYZING with", ollama_model)
    outdir = f'{output_dir}/results_{ollama_model.replace("/", "_").replace(":", "_")}'

    # Create the directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print(f"Created directory {outdir}")
    else:
        print(f"Directory {outdir} already exists.")

    # Set directory permissions to 777
    os.chmod(outdir, 0o777)  # Grant all permissions
    print(f"Set permissions for {outdir}")

    # try:
    ollama_llm = ChatOllama(
        base_url=api_url,
        model=ollama_model,
        temperature=0.0,
        seed=28,
        num_ctx=25000,
        num_predict=-1,
        top_k=100,
        top_p=0.95,
        format="json",
        client_kwargs={'headers': headers})

    # Create academate instance
    screening_ollama = Academate(
        topic=None,
        llm=ollama_llm,
        embeddings=ollama_embeddings,
        criteria_dict=criteria_dict,
        vector_store_path=outdir,
        literature_df=df,
        content_column="Record",
        embeddings_path=f"{output_dir}/ollama_embeddings",
        pdf_location=f"{workdir}/pdfs",
        verbose=False,
        chunksize=25,
        selection_method="all_true"  # Use clustering by default
    )

    # Run screening 1
    results_df = screening_ollama.run_screening1()
    results_df.screening1.value_counts()
    # self = screening_google
    results_df.to_pickle(f'{outdir}/results_screening1.pkl')

    # Run screening 2
    results_df = results_df[results_df['pdf_name'] != 'NO PAPER AVAILABLE.pdf']
    results_df = results_df.copy()  # Create a copy of the dataframe
    results_df.loc[:, 'predicted_screening1'] = results_df['screening1']
    results_df['pdf_path'] = f"{workdir}/pdfs/" + results_df['pdf_name']

    print("\n\LENGTH\n\n", len(results_df), len(set(results_df.uniqueid)))
    print("\n\SCREENING 2\n\n", results_df.screening2.value_counts())

    screening_ollama.results_screening1 = results_df
    # self = screening_google
    results_df2 = screening_ollama.run_screening2()
    # remove rows that are nan in predicted_screening2
    # results_df2 = results_df2.dropna(subset=['predicted_screening2'])
    screening_ollama.generate_excel_report(screening_type='screening2')
    results_df2.to_pickle(f'{outdir}/results_screening2.pkl')
    screening_ollama.create_PRISMA_visualization()

### OPENAI


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("openai_api")
with open("/home/bbb1417/academatepy/validation/openai_model_list.pkl", 'rb') as f:
    openai_models = pickle.load(f)
#
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

for openai_model in openai_models:
    print("\n Analyzing with", openai_model)
    outdir = f'{output_dir}/results_{openai_model}'

    # Create the directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print(f"Created directory {outdir}")
    else:
        print(f"Directory {outdir} already exists.")

    # Set directory permissions to 777
    os.chmod(outdir, 0o777)  # Grant all permissions
    print(f"Set permissions for {outdir}")

    openai_llm = ChatOpenAI(model=openai_model, temperature=0)

    # Create Google AI chat model
    openai_llm = ChatGoogleGenerativeAI(
        model=openai_model,
        temperature=0.0,
        top_p=0.95,
        top_k=40,
        max_output_tokens=2048)

    # Create academate instance
    screening_openai = Academate(
        topic=None,
        llm=openai_llm,
        embeddings=openai_embeddings,
        criteria_dict=criteria_dict,
        vector_store_path=outdir,
        literature_df=df,
        content_column="Record",
        embeddings_path=f"{output_dir}/openai_embeddings",
        pdf_location=f"{workdir}/pdfs",
        verbose=False,
        chunksize=25,
        selection_method="all_true"  # Use clustering by default
    )
    # Run screening 1
    results_df = screening_openai.run_screening1()
    results_df.screening1.value_counts()
    # self = screening_google

    results_df.to_pickle(f'{outdir}/results_screening1.pkl')

    # Run screening 2
    results_df = results_df[results_df['pdf_name'] != 'NO PAPER AVAILABLE.pdf']
    results_df = results_df.copy()  # Create a copy of the dataframe
    results_df.loc[:, 'predicted_screening1'] = results_df['screening1']
    results_df['pdf_path'] = f"{workdir}/pdfs/" + results_df['pdf_name']

    print("\n\LENGTH\n\n", len(results_df), len(set(results_df.uniqueid)))
    print("\n\SCREENING 2\n\n", results_df.screening2.value_counts())

    screening_openai.results_screening1 = results_df
    # self = screening_google
    results_df2 = screening_openai.run_screening2()
    # remove rows that are nan in predicted_screening2
    # results_df2 = results_df2.dropna(subset=['predicted_screening2'])
    screening_openai.generate_excel_report(screening_type='screening2')
    results_df2.to_pickle(f'{outdir}/results_screening2.pkl')
    screening_openai.create_PRISMA_visualization()

### GOOGLE AI
# Set up Google AI API key from environment variable
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("google_api_key")

# List of Google Generative AI models to test
with open("/home/bbb1417/academatepy/validation/google_model_list.pkl", 'rb') as f:
    google_models = pickle.load(f)

# google_models = ["gemini-2.0-flash"]
# Set up Google embeddings
google_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

for google_model in google_models:
    print(f"\nANALYZING with {google_model}")
    outdir = f'{output_dir}/results_{google_model.replace("/", "_").replace(":", "_")}'

    information = {'total_records': len(df),
                   'test_result_type': test_result_type,
                   'model_name': google_model,
                   'reviewname': 'PNP'

                   }

    # Create the directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print(f"Created directory {outdir}")
    else:
        print(f"Directory {outdir} already exists.")

    # Set directory permissions to 777
    os.chmod(outdir, 0o777)  # Grant all permissions
    print(f"Set permissions for {outdir}")

    # Create Google AI chat model
    google_llm = ChatGoogleGenerativeAI(
        model=google_model,
        temperature=0.0,
        top_p=0.95,
        top_k=40,
        max_output_tokens=2048)

    # Create academate instance
    screening_google = Academate(
        topic=None,
        llm=google_llm,
        embeddings=google_embeddings,
        criteria_dict=criteria_dict,
        vector_store_path=outdir,
        literature_df=df,
        content_column="Record",
        embeddings_path=f"{output_dir}/google_embeddings",
        pdf_location=f"{workdir}/pdfs",
        verbose=False,
        chunksize=25,
        selection_method="all_true"  # Use clustering by default
    )
    # Run screening 1
    results_df = screening_google.run_screening1()
    results_df.screening1.value_counts()
    # self = screening_google

    results_df.to_pickle(f'{outdir}/results_screening1.pkl')

    # Run screening 2
    results_df = results_df[results_df['pdf_name'] != 'NO PAPER AVAILABLE.pdf']
    results_df = results_df.copy()  # Create a copy of the dataframe
    results_df.loc[:, 'predicted_screening1'] = results_df['screening1']
    results_df['pdf_path'] = f"{workdir}/pdfs/" + results_df['pdf_name']

    print("\n\LENGTH\n\n", len(results_df), len(set(results_df.uniqueid)))
    print("\n\SCREENING 2\n\n", results_df.screening2.value_counts())

    screening_google.results_screening1 = results_df
    # self = screening_google
    results_df2 = screening_google.run_screening2()
    # remove rows that are nan in predicted_screening2
    # results_df2 = results_df2.dropna(subset=['predicted_screening2'])
    screening_google.generate_excel_report(screening_type='screening2')
    results_df2.to_pickle(f'{outdir}/results_screening2.pkl')
    screening_google.create_PRISMA_visualization()


def diagnose_screening_state(academate_instance):
    """Diagnose the current state of screening"""

    # Check screening1 state
    print("\nSCREENING 1 STATE:")
    print(f"Record2Answer entries: {len(academate_instance.title_abstract_screener.record2answer)}")
    print(f"Missing records: {len(academate_instance.title_abstract_screener.missing_records)}")

    # Check screening2 state
    print("\nSCREENING 2 STATE:")
    print(f"Record2Answer entries: {len(academate_instance.full_text_screener.record2answer)}")
    print(f"Missing records: {len(academate_instance.full_text_screener.missing_records)}")
    print(f"PDF embedding errors: {len(academate_instance.pdf_embedder.pdf_embedding_error)}")

    # Check database state
    embeddings_path = academate_instance.embeddings_path2
    if os.path.exists(os.path.join(embeddings_path, 'chroma.sqlite3')):
        print("\nDatabase exists at:", embeddings_path)
        try:
            from langchain_chroma import Chroma
            db = Chroma(
                collection_name="screening2",
                persist_directory=embeddings_path,
                embedding_function=academate_instance.embeddings
            )
            print(f"Database contains {db._collection.count()} records")
        except Exception as e:
            print(f"Error checking database: {e}")
    else:
        print("\nNo database found at:", embeddings_path)


# Call this after each main operation
diagnose_screening_state(screening_google)
