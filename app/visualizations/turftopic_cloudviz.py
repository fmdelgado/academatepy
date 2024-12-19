from turftopic import ClusteringTopicModel
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from langchain_ollama.chat_models import ChatOllama
import requests, json
from dotenv import load_dotenv
from app.visualizations.custom_topic_namer import LocalTopicNamer
import os
import pandas as pd

dotenv_path = '/Users/fernando/Documents/Research/academatepy/.env'
load_dotenv(dotenv_path)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create and fit the model with ONLY the feature_importance parameter
from sklearn.datasets import fetch_20newsgroups

cuisine_docs = [
    "I love making homemade pasta and experimenting with new sauces.",
    "The best part of baking bread is the aroma filling the kitchen.",
    "Culinary arts require a delicate balance of flavors and techniques.",
    "Fresh ingredients are the key to a delicious meal.",
    "Cooking with spices transforms simple dishes into gourmet experiences."
]

science_docs = [
    "Quantum mechanics explores the behavior of particles at the smallest scales.",
    "The theory of relativity revolutionized our understanding of time and space.",
    "Evolution by natural selection explains biodiversity.",
    "The human genome project unlocked the blueprint of life.",
    "Climate change science warns us about human impact on the planet."
]

sports_docs = [
    "Football is a sport that brings people together.",
    "Athletes train rigorously to improve their performance.",
    "The Olympics showcase the best talent from around the world.",
    "Tennis matches require both stamina and precision.",
    "Basketball involves strategy, speed, and teamwork."
]

music_docs = [
    "Classical music soothes the soul and sharpens the mind.",
    "Jazz improvisation showcases musicians' creativity.",
    "Rock concerts excite audiences with energetic performances.",
    "Pop music often dominates the airwaves and playlists.",
    "Opera combines vocal artistry with dramatic storytelling."
]



corpus = cuisine_docs + science_docs + sports_docs + music_docs
test = pd.read_pickle("/Users/fernando/Documents/Research/academatepy/validation/AI_healthcare/preprocessed_articles_filtered.pkl")

corpus = test['Record'].to_list()
# Fit the model
model = ClusteringTopicModel(
    feature_importance="centroid",
    vectorizer=CountVectorizer(min_df=1),
    clustering=HDBSCAN(min_cluster_size=5, min_samples=1)  # Adjust these as needed
)
model.fit(corpus)

# Create and apply the namer
# namer = OpenAITopicNamer("gpt-4o-mini")

auth_url = os.getenv("AUTH_URL")
api_url = os.getenv('OLLAMA_API_URL')
account = {
    'email': os.getenv("ollama_user"),
    'password': os.getenv("ollama_pw")
}


auth_response = requests.post(auth_url, json=account)
jwt = auth_response.json()["token"]
headers = {"Authorization": "Bearer " + jwt}
ollama_model = 'llama3.3:70b'

ollama_llm_json = ChatOllama(
            base_url=api_url,
            model=ollama_model,
            temperature=0.0,
            seed=28,
            num_ctx=25000,
            num_predict=-1,
            # top_k=100,
            # top_p=0.95,
            # format="json",
            client_kwargs={'headers': headers})


namer = LocalTopicNamer(llm=ollama_llm_json)
model.rename_topics(namer)

# Generate the visualization
fig = model.plot_clusters_datamapplot()
fig.save("clusters_visualization.html")

import topicwizard

topicwizard.visualize(model=model, corpus=corpus)