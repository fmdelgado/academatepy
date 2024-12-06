# ### Step 1: Import Necessary Libraries
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
import requests
import os
from dotenv import load_dotenv
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
sys.path.append("/Users/fernando/Documents/Research/academatepy/app")
from academate_new import academate
import nest_asyncio

dotenv_path = '/Users/fernando/Documents/Research/academatepy/.env'
load_dotenv(dotenv_path)

## MODEL SELECTOR
# Authentication details

auth_url = os.getenv("AUTH_URL")
api_url = os.getenv('OLLAMA_API_URL')
account = {
    'email': os.getenv("ollama_user"),
    'password': os.getenv("ollama_pw")
}


auth_response = requests.post(auth_url, json=account)
jwt = auth_response.json()["token"]
headers = {"Authorization": "Bearer " + jwt}


# ### Step 2: Load the DataFrame
# Define the working directory and load the dataset.
workdir = "/Users/fernando/Documents/Research/academatepy/app/test"  # Update with your working directory.
df_path = f"{workdir}/test_data.pkl"  # Replace with your dataset file.

# Load the dataset
df = pd.read_pickle(df_path)


criteria_dict = {
    "AI_functionality_description": "Return true if the text provides a comprehensive description of an AI functionality used in healthcare; otherwise, return false.",

    "Economic_evaluation": " Return true if the text evaluates the economic efficiency and outcomes of an AI application in healthcare, specifically assessing cost-effectiveness or return on investment; otherwise, return false.",

    "Quantitative_healthcare_outcomes": "Return true if the text reports quantitative outcomes in at least one healthcare system, showing measurable impacts such as patient recovery times, treatment efficacy, or cost savings; otherwise, return false.",

    "Relevance_AI_Healthcare": "Return false if the text does not explicitly cover a topic related to AI in healthcare, indicating the study is not primarily focused on AI applications within healthcare; otherwise, return true.",

    "AI_application_description": "Return false if the abstract or full text does not contain a description of an AI application in healthcare, indicating a lack of focus on how AI technologies are implemented or their functional roles within healthcare; otherwise, return true.",

    "Economic_outcome_details": "Return false if the abstract or full text does not elaborate on the quantitative economic outcomes in one healthcare system, failing to provide specific economic data or analysis related to the AI application; otherwise, return true."
}

ollama_model = 'mistral:v0.2'

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

ollama_embeddings = OllamaEmbeddings(base_url=api_url, model="nomic-embed-text",
                                     headers={"Authorization": "Bearer " + jwt})

output_dir = f"{workdir}/output"
# Create an instance of the Academate class
screening_ollama = academate(topic=None, llm=ollama_llm, embeddings=ollama_embeddings,
                             criteria_dict=criteria_dict, vector_store_path=output_dir, literature_df=df,
                             content_column="Record", embeddings_path=f"{output_dir}/ollama_embeddings",
                             pdf_location=f"{output_dir}/pdfs", verbose=False, chunksize=25)

results_df = screening_ollama.run_screening1()
self = screening_ollama
results_df2 = screening_ollama.run_screening2()
