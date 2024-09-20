import os
from dotenv import load_dotenv
import requests, json
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama

dotenv_path = '/Users/fernando/Documents/Research/academatepy/.env'
load_dotenv(dotenv_path)

auth_response = requests.post(os.getenv('AUTH_URL'), json= {'email': os.getenv('ollama_user'), 'password': os.getenv('ollama_pw')})
jwt = json.loads(auth_response.text)["token"]
headers = {"Authorization": "Bearer " + jwt}

ollama_embeddings = OllamaEmbeddings(base_url=os.getenv('OLLAMA_API_URL'), model="nomic-embed-text", headers=headers)
ollama_llm = ChatOllama(base_url=os.getenv('OLLAMA_API_URL'), model= 'reflection:70b', temperature=0.0,
                        client_kwargs={'headers': headers}, format="json")
messages = [("system", "You are a helpful trip advisor."), ("human", "What can we visit in Hamburg?"), ("assistant","blablabla"),("human", "How about dark tourism?")]
response = ollama_llm.invoke(messages)
print(response)
