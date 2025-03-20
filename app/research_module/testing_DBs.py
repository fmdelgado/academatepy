from langchain_ollama.chat_models import ChatOllama
import requests, json
from dotenv import load_dotenv
import os
dotenv_path = '/Users/fernando/Documents/Research/academatepy/.env'
load_dotenv(dotenv_path)
import sys
## MODEL SELECTOR
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


ollama_model = "nemotron:latest"

ollama_llm = ChatOllama(
    base_url=api_url,
    model=ollama_model,
    temperature=0.0,
    seed=28,
    num_ctx=25000,
    num_predict=-1,
    top_k=100,
    top_p=0.95,
    client_kwargs={'headers': headers})

import sys
sys.path.append("/Users/fernando/Documents/Research/academatepy/app/research_module")

topic = "The role of artificial intelligence in drug discovery"

model_list = [
 'granite3-dense:8b',
 'granite3-moe:latest',
 'nemotron:latest',
 'nemotron-mini:4b',
 'mistral:v0.2',
 'llama3.2:1b',
 'llama3.2:latest',
 'qwen2.5:latest',
 'mistral:7b',
 'finalend/athene-70b:latest',
 'mixtral:8x7b',
 'reflection:70b',
 'reflection:latest',
 'gemma2:27b',
 'llama3:8b-instruct-fp16',
 'llama3:latest',
 'gemma:latest',
 'mixtral:8x22b',
 'phi3:latest',
 'phi3:14b',
 'phi3:3.8b',
 'mistral-nemo:latest',
 'medllama2:latest',
 'meditron:70b',
 'meditron:7b',
 'llama3.1:8b',
 'llama3.1:70b',
 'gemma2:9b']

## PUBMED QUERYING
# Create an instance of the PubMedQueryGenerator
from pubmed_querying import PubMedQueryGenerator
email = "your.email@example.com"
pubmed_api_key = os.getenv("NCBI_API_KEY")  # Replace with your actual NCBI API key

# pubmed_query_gen = PubMedQueryGenerator(email, pubmed_api_key, ollama_llm)
# Optionally set date filters
# pubmed_query_gen.set_date_filter(from_year=2010)
#
# # Generate a query for a topic
# query = pubmed_query_gen.generate_query(topic)
#
# print(f"Generated Query:\n{query}")
# # Execute the query
# pubmed_query_gen.execute_query(query)
# # Get the results as a DataFrame
# results_df = pubmed_query_gen.results

# for model_name in model_list:
#     print(f"Testing model: {model_name}")
#     # Create an instance of the LLM with the current model
#     ollama_llm = ChatOllama(
#         base_url=api_url,
#         model=model_name,
#         temperature=0.0,
#         seed=28,
#         num_ctx=25000,
#         num_predict=-1,
#         top_k=100,
#         top_p=0.95,
#         client_kwargs={'headers': headers}
#     )
#
#     # Create a new instance of the PubMedQueryGenerator
#     pubmed_query_gen = PubMedQueryGenerator(email, pubmed_api_key, ollama_llm)
#     pubmed_query_gen.set_date_filter(from_year=2010)
#
#     # Generate the query
#     try:
#         query = pubmed_query_gen.generate_query(topic)
#         print(f"Generated Query with {model_name}:\n{query}\n")
#
#         # Execute the query
#         pubmed_query_gen.execute_query(query)
#         results_df = pubmed_query_gen.get_results_dataframe()
#         if not results_df.empty:
#             print(f"Results with {model_name}:\n{len(results_df)}\n")
#         else:
#             print(f"No results found with {model_name}.\n")
#     except Exception as e:
#         print(f"Error with model {model_name}: {e}\n")




## EMBASE QUERYING
from embase_querying import EmbaseQueryGenerator
email = "your.email@example.com"
embase_api_key = os.getenv("EMBASE_API_KEY")  # Replace with your actual Embase API key
# Generate a query for a topic
# embase_query_gen = EmbaseQueryGenerator(embase_api_key, ollama_llm)
# embase_query_gen.set_date_filter(from_year=2010)
# query = embase_query_gen.generate_embase_query(topic)
# print(f"Generated Query with {ollama_model}:\n{query}\n")
# # Optionally, you can test executing the query and handle exceptions
# try:
#     embase_query_gen.execute_query(query)
#     results_df = embase_query_gen.get_results_dataframe()
#     print(f"Results with {ollama_model}:\n{results_df.head()}\n")
# except Exception as e:
#     print(f"Error with model {ollama_model}: {e}\n")


# for model_name in model_list:
#     print(f"Testing model: {model_name}")
#     # Create an instance of the LLM with the current model
#     ollama_llm = ChatOllama(
#         base_url=api_url,
#         model=model_name,
#         temperature=0.0,
#         seed=28,
#         num_ctx=25000,
#         num_predict=-1,
#         top_k=100,
#         top_p=0.95,
#         format="json",
#         client_kwargs={'headers': headers})
#
#     # Create a new instance of the EmbaseQueryGenerator
#     embase_query_gen = EmbaseQueryGenerator(embase_api_key, ollama_llm)
#     embase_query_gen.set_date_filter(from_year=2010)
#
#     # Generate the query
#     try:
#         query = embase_query_gen.generate_query(topic)
#         print(f"Generated Query with {model_name}:\n{query}\n")
#
#         # Execute the query
#         embase_query_gen.execute_query(query)
#         results_df = embase_query_gen.get_results_dataframe()
#         if not results_df.empty:
#             print(f"Results with {model_name}:\n{len(results_df)}\n")
#         else:
#             print(f"No results found with {model_name}.\n")
#     except Exception as e:
#         print(f"Error with model {model_name}: {e}\n")
#
#

# # Import the ScopusQueryGenerator
# # # Import the ScopusQueryGenerator
# from scopus_querying import ScopusQueryGenerator
# #
# # # Your Scopus API key
# scopus_api_key = os.getenv("SCOPUS_API_KEY")  # Replace with your actual Scopus API key
#
# # Your research topic
# topic = "The role of artificial intelligence in drug discovery"
#
# for model_name in model_list:
#     print(f"Testing model: {model_name}")
#     # Create an instance of the LLM with the current model
#     ollama_llm = ChatOllama(
#         base_url=api_url,
#         model=model_name,
#         temperature=0.0,
#         seed=28,
#         num_ctx=25000,
#         num_predict=-1,
#         top_k=100,
#         top_p=0.95,
#         client_kwargs={'headers': headers}
#     )
#
#     # Create a new instance of the ScopusQueryGenerator
#     scopus_query_gen = ScopusQueryGenerator(scopus_api_key, ollama_llm)
#     scopus_query_gen.set_date_filter(from_year=2010)
#
#     # Generate the query
#     try:
#         query = scopus_query_gen.generate_query(topic)
#         print(f"Generated Query with {model_name}:\n{query}\n")
#
#         # Execute the query
#         scopus_query_gen.execute_query(query)
#         results_df = scopus_query_gen.get_results_dataframe()
#         if not results_df.empty:
#             print(f"Results with {model_name}:\n{len(results_df)}\n")
#         else:
#             print(f"No results found with {model_name}.\n")
#     except Exception as e:
#         print(f"Error with model {model_name}: {e}\n")
#

#
# # Import the SemanticScholarQueryGenerator
# from sscholar_querying import SemanticScholarQueryGenerator
#
# # Your Semantic Scholar API key
semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")  # Replace with your actual API key
#
# # Your research topic
# topic = "The role of artificial intelligence in drug discovery"
#
# for model_name in model_list:
#     print(f"Testing model: {model_name}")
#     # Create an instance of the LLM with the current model
#     ollama_llm = ChatOllama(
#         base_url=api_url,
#         model=model_name,
#         temperature=0.0,
#         seed=28,
#         num_ctx=25000,
#         num_predict=-1,
#         top_k=100,
#         top_p=0.95,
#         client_kwargs={'headers': headers}
#     )
#
#     # Create a new instance of the SemanticScholarQueryGenerator
#     ss_query_gen = SemanticScholarQueryGenerator(semantic_scholar_api_key, ollama_llm)
#     ss_query_gen.set_date_filter(from_year=2010)
#
#     # Generate the query
#     try:
#         query = ss_query_gen.generate_query(topic)
#         print(f"Generated Query with {model_name}:\n{query}\n")
#
#         # Execute the query
#         ss_query_gen.execute_query(query)
#         results_df = ss_query_gen.get_results_dataframe()
#         if not results_df.empty:
#             print(f"Results with {model_name}:\n{len(results_df)}\n")
#         else:
#             print(f"No results found with {model_name}.\n")
#     except Exception as e:
#         print(f"Error with model {model_name}: {e}\n")
#


from literature_searcher import UnifiedLiteratureSearcher
# Instantiate the UnifiedLiteratureSearcher with the necessary API keys and LLM instance
searcher = UnifiedLiteratureSearcher(
    pubmed_email='your_email@example.com',
    pubmed_api_key=pubmed_api_key,
    scopus_api_key=None,  # Explicitly pass None
    semantic_scholar_api_key=semantic_scholar_api_key,
    llm=ollama_llm,
    from_year=2010,
    to_year=2023
)

topic="The role of artificial intelligence in drug discovery"
# Perform the search
# self = searcher

searcher.search(topic="The role of artificial intelligence in drug discovery")

# Retrieve the unified results
results_df = searcher.get_results_dataframe()

# View the queries used
queries = searcher.get_queries()

# Access individual database results
db_results = searcher.get_database_results()

# Loop over the models in model_list
for model_name in model_list:
    print(f"Testing model: {model_name}")

    # Create an instance of the LLM with the current model
    ollama_llm = ChatOllama(
        base_url=api_url,
        model=model_name,
        temperature=0.0,
        seed=28,
        num_ctx=25000,
        num_predict=-1,
        top_k=100,
        top_p=0.95,
        client_kwargs={'headers': headers}
    )

    # Create an instance of UnifiedLiteratureSearcher with the LLM and API keys
    searcher = UnifiedLiteratureSearcher(
        pubmed_email="test@mailinator.com",
        pubmed_api_key=pubmed_api_key,
        scopus_api_key=None,
        semantic_scholar_api_key=semantic_scholar_api_key,
        llm=ollama_llm,
        from_year=2010,
        to_year=2024
    )

    # Perform the search
    try:
        searcher.search(topic=topic)

        # Retrieve the unified results
        results_df = searcher.get_results_dataframe()

        # Print the number of results
        if not results_df.empty:
            print(f"Results with {model_name}:\n{len(results_df)} articles retrieved.\n")
        else:
            print(f"No results found with {model_name}.\n")

        # Optionally, print the queries used
        queries = searcher.get_queries()
        print(f"Queries used with {model_name}:")
        for db, query in queries.items():
            print(f"{db} Query:\n{query}\n")

    except Exception as e:
        print(f"Error with model {model_name}: {e}\n")