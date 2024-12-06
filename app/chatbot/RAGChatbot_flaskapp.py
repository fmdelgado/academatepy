from RAGChatbot import RAGChatbot  # Replace 'your_module' with the actual module name
from flask import Flask, request, jsonify
import os
import json
import requests
from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Load environment variables
dotenv_path = '/Users/fernando/Documents/Research/academatepy/.env'
load_dotenv(dotenv_path)

# Authentication (if needed)
auth_response = requests.post(
    os.getenv('AUTH_URL'),
    json={'email': os.getenv('ollama_user'), 'password': os.getenv('ollama_pw')}
)
jwt = json.loads(auth_response.text)["token"]
headers = {"Authorization": "Bearer " + jwt}

# Initialize LLM
llm = ChatOllama(
    base_url=os.getenv('OLLAMA_API_URL'),
    model='mistral:7b',
    temperature=0.1,
    num_predict=-1,
    seed=20,
    stop=["Answer:"],
    client_kwargs={'headers': headers}
)

# Initialize embeddings
embeddings = OllamaEmbeddings(
    base_url=os.getenv('OLLAMA_API_URL'),
    model="nomic-embed-text",
    headers=headers
)

# Load vectorstore
persist_directory = "/Users/fernando/Documents/Research/academatepy/test_results/results_llama3.1_8b/embeddings/screening1_embeddings"
vectorstore = Chroma(
    collection_name="literature",
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# Initialize the chatbot
chatbot = RAGChatbot(
    llm=llm,
    embeddings=embeddings,
    chromadb=vectorstore
)

# Instantiate chatbot
chatbot = RAGChatbot(
    llm=llm,
    embeddings=embeddings,
    chromadb=vectorstore
)

# Dictionary to store conversation histories for each user
conversation_histories = {}


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message')
    user_id = data.get('user_id', 'default_user')

    if not user_input:
        return jsonify({'error': 'No input provided.'}), 400

    # Retrieve or initialize conversation history
    conversation_history = conversation_histories.get(user_id, [])

    # Get the answer from the chatbot
    answer, source_mapping = chatbot.get_answer(user_input, conversation_history, user_id)

    # Update the conversation history
    conversation_histories[user_id] = conversation_history

    # Prepare the response
    response = {
        'answer': answer,
        'sources': []
    }

    if source_mapping:
        for source_id, doc in source_mapping.items():
            response['sources'].append({
                'source_id': source_id,
                'metadata': doc.metadata,
                'content': doc.page_content
            })

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)