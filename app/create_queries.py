# Run: python create_queries.py "{topic}"

# ### Step 1: Import Necessary Libraries
from research_module.literature_searcher import UnifiedLiteratureSearcher
from langchain_ollama.chat_models import ChatOllama
from dotenv import load_dotenv
import requests
import os

import sys
sys.path.append('.')
dotenv_path = '../.env'
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

model_name = 'llama3.1:8b'

ollama_llm = ChatOllama(
    base_url=api_url,
    model=model_name,
    temperature=0.0,
    seed=28,
    num_ctx=25000,
    num_predict=-1,
    top_k=100,
    top_p=0.95,
    client_kwargs={'headers': headers}, 
)

topic = sys.argv[1] #"The role of artificial intelligence in drug discovery"

# Create an instance of UnifiedLiteratureSearcher with the LLM and API keys

pubmed_api_key = os.getenv("NCBI_API_KEY")  
semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") 
scopus_api_key = os.getenv("SCOPUS_API_KEY")

searcher = UnifiedLiteratureSearcher(
    pubmed_email=os.getenv("ollama_user"),
    pubmed_api_key=pubmed_api_key,
    scopus_api_key=scopus_api_key,
    semantic_scholar_api_key=semantic_scholar_api_key,
    llm=ollama_llm,
    from_year=2010,
    to_year=2024
)
try:
    searcher.search(topic=topic)
    # Retrieve the unified results
    literature_df = searcher.get_results_dataframe()
    # Print the number of results
    if not literature_df.empty:
        print(f"Results with {model_name}:\n{len(literature_df)} articles retrieved.\n")
    else:
        print(f"No results found with {model_name}.\n")
    # Print the queries used
    queries = searcher.get_queries()
    print(f"Queries used with {model_name}:")
    for db, query in queries.items():
        print(f"{db} Query:\n{query}\n")
except Exception as e:
    print(f"Error with model {model_name}: {e}\n")
    