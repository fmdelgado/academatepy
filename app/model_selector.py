import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
import requests, json
from dotenv import load_dotenv
import os

protocol = "https"
hostname = "chat.cosy.bio"

host = f"{protocol}://{hostname}"

auth_url = f"{host}/api/v1/auths/signin"
api_url = f"{host}/ollama"

# screen -S llm_pnp python analyze_PNP.py
dotenv_path = '/Users/fernando/Documents/Research/academatepy/.env'
load_dotenv(dotenv_path)


account = {
    'email': os.getenv("ollama_user"),
    'password': os.getenv("ollama_pw")
}
auth_response = requests.post(auth_url, json=account)
jwt = json.loads(auth_response.text)["token"]

headers = {"Authorization": f"Bearer {jwt}"}
response = requests.get(f"{api_url}/api/tags", headers=headers)
if response.status_code == 200:
    models = response.json()

# Assuming 'models' is your JSON object loaded from a response
models_data = models['models']
models_below_10b = []

# Loop through each model in the JSON data
for model in models_data:
    # Checking if the model might be a language generative model based on details description
    # This is a very basic heuristic and might need refinement depending on your actual data
    if 'language' in model['details'].get('format', '').lower() or 'llama' in model['details'].get('family',
                                                                                                   '').lower() or 'bert' in \
            model['details'].get('family', '').lower() or 'gpt' in model['details'].get('family', '').lower():
        # Check if 'parameter_size' exists and if it contains 'B' indicating billion
        if 'parameter_size' in model['details'] and 'B' in model['details']['parameter_size']:
            # Extract the number of parameters as a float and check if it's less than 10 billion
            param_size = float(model['details']['parameter_size'].replace('B', ''))
            if param_size < 10:
                models_below_10b.append(model['name'])

df = pd.DataFrame(models_data)
df = df.join(pd.json_normalize(df['details']))
df.name.to_list()




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

ollama_model = "mixtral:8x22b"
ollama_llm_json = ChatOllama(
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

ollama_llm = ChatOllama(
    base_url=api_url,
    model=ollama_model,
    temperature=0.0,
    seed=28,
    num_ctx=25000,
    num_predict=-1,
    top_k=100,
    top_p=0.95,
    # format="json",
    client_kwargs={'headers': headers})

messages = "hello! What can you recommend in Hamburg?"
response = ollama_llm.invoke(messages)
print(response.content)
