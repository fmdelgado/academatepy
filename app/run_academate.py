import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
import requests, json
from dotenv import load_dotenv
import os
import sys

sys.path.append("/Users/fernando/Documents/Research/academatepy/app")
from academate_new import academate
import pickle

# screen -S llm_AI_chroma_chatcosy_fixedparams python new_analysis_AI_fixedparams.py

dotenv_path = '/Users/fernando/Documents/Research/academatepy/.env'
load_dotenv(dotenv_path)

# READING THE DF
df = pd.read_pickle(f'/Users/fernando/Documents/Research/academatepy/validation/AI_healthcare/preprocessed_articles_filtered.pkl')
output_dir = "/Users/fernando/Documents/Research/academatepy/app/test"

print(df.columns)

criteria_dict = {
    "AI_functionality_description": "Return true if the study provides a comprehensive description of an AI functionality used in healthcare; otherwise, return false.",

    "Economic_evaluation": " Return true if the study evaluates the economic efficiency and outcomes of an AI application in healthcare, specifically assessing cost-effectiveness or return on investment; otherwise, return false.",

    "Quantitative_healthcare_outcomes": "Return true if the study reports quantitative outcomes in at least one healthcare system, showing measurable impacts such as patient recovery times, treatment efficacy, or cost savings; otherwise, return false.",

    "Relevance_AI_Healthcare": "Return false if the title of the study does not explicitly cover a topic related to AI in healthcare, indicating the study is not primarily focused on AI applications within healthcare; otherwise, return true.",

    "AI_application_description": "Return false if the abstract does not contain a description of an AI application in healthcare, indicating a lack of focus on how AI technologies are implemented or their functional roles within healthcare; otherwise, return true.",

    "Economic_outcome_details": "Return false if the abstract or full text does not elaborate on the quantitative economic outcomes in one healthcare system, failing to provide specific economic data or analysis related to the AI application; otherwise, return true."
}

# Read a pickle file
with open("/Users/fernando/Documents/Research/academatepy/validation/ollama_model_list.pkl", 'rb') as f:
    ollama_models = pickle.load(f)
# If you want to see what's in it

ollama_models = ['llama3.2:1b',
                 'llama3.2:latest',
                 'mistral:v0.2']

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

ollama_embeddings = OllamaEmbeddings(base_url=api_url, model="nomic-embed-text",
                                     headers={"Authorization": "Bearer " + jwt})
ollama_model= "mistral:v0.2"


for ollama_model in ollama_models:
    print("\n Analyzing with", ollama_model)
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

    try:
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

        screening_ollama = academate(topic=None, llm=ollama_llm, embeddings=ollama_embeddings,
                                     criteria_dict=criteria_dict, vector_store_path=outdir, literature_df=df,
                                     content_column="Record", embeddings_path=f"{output_dir}/ollama_embeddings",
                                     pdf_location=f"{output_dir}/pdfs", verbose=True, chunksize=25)

        results_df = screening_ollama.run_screening1()
        results_df.screening1.value_counts()
        screening_ollama.generate_excel_report(screening_type='screening1')

        results_df['predicted_screening1'] = results_df['screening1']
        results_df['predicted_screening1'].value_counts()
        results_df['pdf_path'] = "/Users/fernando/Documents/Research/academatepy/app/test/pdfs/" + results_df['pdf_name']

        screening_ollama.results_screeening1 = results_df
        screening_ollama.results_screeening1['predicted_screening1'].value_counts()
        # self = screening_ollama
        results_df = screening_ollama.run_screening2()
        screening_ollama.generate_excel_report(screening_type='screening2')
        results_df.to_pickle(f'{outdir}/results_screening2.pkl')

        prisma_fig = screening_ollama.generate_prisma_flow_diagram()
        prisma_fig.write_image(f"{outdir}/prisma_flow_diagram.png")
        prisma_fig.write_image(f"{outdir}/prisma_flow_diagram.html")
    except:
        pass



