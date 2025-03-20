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
sys.path.append("/Users/fernando/Documents/Research/academatepy/")
from academate import AcademateSingleDB as Academate  # This imports the standard implementation
import pickle
import numpy as np

# screen -S llm_reproduction_academate python analyze_repro_modular.py
test_result_type = "results"

dotenv_path = '/Users/fernando/Documents/Research/academatepy/.env'
load_dotenv(dotenv_path)

# READING THE DF
workdir = "/Users/fernando/Documents/Research/academatepy/validation/reproduction"
df = pd.read_pickle(f'/Users/fernando/Documents/Research/academatepy/validation/reproduction/preprocessed_articles_filtered.pkl')
output_dir = f"{workdir}/{test_result_type}"

print(df.columns)

criteria_dict = {
    "Population": "If the study population includes humans with endometrial disorders (such as Asherman's syndrome, intrauterine adhesions, endometrial atrophy, thin endometrium, or endometritis), animals modeling endometrial disorders  (such as Asherman's syndrome, intrauterine adhesions, endometrial atrophy, thin endometrium, or endometritis), or in vitro studies modeling endometrial disorders  (such as Asherman's syndrome, intrauterine adhesions, endometrial atrophy, thin endometrium, or endometritis)  return True. If the study only involves healthy humans without endometrial disorders, animal models not related with endometrial disorders, or in vitro studies not related with endometrial disorders, return False. If the study exclusively involves human with endometriosis, RIF, cancer, or adenomyosis, return False. If the study exclusively involves animal models or in vitro models of endometriosis, RIF, cancer, or adenomyosis, return False.",
    "Intervention": "If the study evaluates a regenerative therapy, including cellular therapies (using stem cells like MSCs, BMMSCs, EndoMSCs, UCMSCs, MenMSCs, AdiMSCs, iPSCs, AMSCs, organoids or embryonic stem cells), acellular therapies (like PRP, EVs, miRNAs, conditioned medium, G-CSF, mitochondria, or apoptotic bodies), or bioengineering approaches (using scaffolds, hydrogels, dECM, or other biomaterials), return True. If the study exclusively evaluates pharmacological treatments (like estrogen, hCG, GnRH, growth hormone, aspirin, sildenafil, pentoxifylline, l-arginine, nitroglycerine, or tocopherol) without a regenerative or bioengineering component, return False. If the intervention is a combination therapy that includes a regenerative or bioengineering approach (e.g., cells + scaffold, PRP + hydrogel), return True. Otherwise return False.",
    "Human": "If the cellular or acellular therapy is derived from human tissues (bone marrow, endometrium, umbilical cord, menstrual blood, adipose tissue, placenta, amniotic fluid, human dECM, etc.) or human blood, return True. If the therapy is derived from exclusively animal sources without any human component , return False. If the study uses animal-derived biomaterials but combines them with human cells or factors, return True. Otherwise retulr False.",
    "Preclinical_Clinical": "If the study is a preclinical study (in vitro or in vivo animal models) or a clinical trial involving human patients, return True. If the study is a review, opinion article, technical article, editorial, letter to the editor, personal opinion, book, book chapter, or untranslated document, return False.",
    "Outcome": "If the study reports outcomes related to endometrial regeneration, repair, or function, including (but not limited to) endometrial thickness, uterine glands, expression of proliferation markers, fibrosis, regenerative markers, AFS score, menstrual changes, or fertility outcomes, wound healing, presence of growth factors, angiogenesis, immunotolerance, cell viability, cell apoptosis, return True. If the study only reports outcomes unrelated to endometrial regeneration (e.g., purely biochemical assays without a link to tissue repair), return False.",
    "Publicationtype": "If the article is an original peer-reviewed full-text article, return True. If the article is a review, opinion piece, technical note, editorial, letter, personal opinion, book, book chapter, or untranslated to English document, return False. If there is no abstract available, return False."
}

### GOOGLE AI
# Set up Google AI API key from environment variable
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("google_api_key")

# List of Google Generative AI models to test
with open("/Users/fernando/Documents/Research/academatepy/validation/google_model_list.pkl", 'rb') as f:
    google_models = pickle.load(f)

# google_models = ["gemini-2.0-flash"]
# Set up Google embeddings
google_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
# google_models = ["gemini-2.0-flash"]

for google_model in google_models:
    print(f"\nANALYZING with {google_model}")
    outdir = f'{output_dir}/results_{google_model.replace("/", "_").replace(":", "_")}'

    information = {'total_records': len(df),
                   'test_result_type': test_result_type,
                   'model_name': google_model,
                   'reviewname': 'reproduction'

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
    results_df = results_df[~results_df['pdf_name'].fillna('').str.contains('no_paper')]
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



### OPENAI
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("openai_api")
with open("/Users/fernando/Documents/Research/academatepy/validation/openai_model_list.pkl", 'rb') as f:
    openai_models = pickle.load(f)
# openai_models = ['gpt-3.5-turbo-0125']
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


### OLLAMA

# Read a pickle file
with open("/Users/fernando/Documents/Research/academatepy/validation/ollama_model_list.pkl", 'rb') as f:
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
ollama_models = ['llama3.3:70b']

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

    # answer = ollama_llm.invoke(['hi, how are you?'])
    # answer.content
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
    results_df = results_df[~results_df['pdf_name'].fillna('').str.contains('no_paper')]
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

