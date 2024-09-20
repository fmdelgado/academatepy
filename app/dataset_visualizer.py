import pandas as pd
import umap.umap_ as umap
import re
from typing import Any
import graphviz as gv
from langchain_community.vectorstores import FAISS
import numpy as np
import faiss
import umap
import hdbscan
import plotly.express as px
# from plotly.offline import plot
import plotly.graph_objects as go
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence


class DatasetVisualizer:
    def __init__(self, llm):
        self.llm = llm

    def plot_vector_DB(self, faiss_db):
        """
        Plot the vector database.
        Args:
            faiss_db:
            llm:

        Returns:
            Plotly figure.
        """
        embedding_vectors, metadata = self.get_all_embeddings_and_metadata(faiss_db)
        # pd.DataFrame.from_records(metadata)
        umap_embeddings = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2).fit_transform(embedding_vectors)
        labels = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, gen_min_span_tree=True).fit_predict(
            umap_embeddings)
        metadata['Cluster'] = labels

        # Create a dictionary to store titles for each cluster
        cluster_titles = {}
        for label, title in zip(labels, metadata['title']):
            if label not in cluster_titles:
                cluster_titles[label] = []
            cluster_titles[label].append(title)

        # Generate cluster names using LLM
        cluster_names = {}
        for label, titles in cluster_titles.items():
            if label == -1:
                cluster_names[label] = 'Outlier'
            else:
                cluster_names[label] = self._generate_cluster_name(self.llm, titles)

        plot_data = pd.DataFrame({
            'x': umap_embeddings[:, 0],
            'y': umap_embeddings[:, 1],
            # 'Cluster':  metadata['Cluster'],
            'Cluster': [cluster_names[label] for label in labels],
            'Title': metadata['title'],
        })

        fig = px.scatter(plot_data, x='x', y='y', color='Cluster', hover_data=['Title', ],
                         color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(title='UMAP of documents clustered by semantic similarity', legend_title='Clusters')
        return fig

    def visualize_PRISMA(self, df):
        # Fill NaN values with 'None'
        df.fillna('None', inplace=True)

        # Define labels for the nodes
        labels = [
            "Total Articles",
            "Analyzed Screening 1 - Yes", "Analyzed Screening 1 - No",
            "Predicted Screening 1 - Yes", "Predicted Screening 1 - No",
            "PDF Downloaded - Yes", "PDF Downloaded - No",
            "PDF Embedded - Yes", "PDF Embedded - No",
            "Predicted Screening 2 - Yes", "Predicted Screening 2 - No"
        ]

        # Initialize source, target, and value lists
        sources, targets, values = [], [], []
        label_annotations = {label: 0 for label in labels}

        def add_flow(source_label, target_label, count):
            if count > 0:
                source_index = labels.index(source_label)
                target_index = labels.index(target_label)
                sources.append(source_index)
                targets.append(target_index)
                values.append(count)
                label_annotations[target_label] += count

        # Total articles
        total_articles = len(df)
        label_annotations["Total Articles"] = total_articles

        # Analyzed Screening 1
        analyzed_screening1_yes = df[df['analyzed_screening1'] == True]
        analyzed_screening1_no = df[df['analyzed_screening1'] == False]
        add_flow("Total Articles", "Analyzed Screening 1 - Yes", len(analyzed_screening1_yes))
        add_flow("Total Articles", "Analyzed Screening 1 - No", len(analyzed_screening1_no))

        # Predicted Screening 1
        predicted_screening1_yes = analyzed_screening1_yes[analyzed_screening1_yes['predicted_screening1'] == True]
        predicted_screening1_no = analyzed_screening1_yes[analyzed_screening1_yes['predicted_screening1'] == False]
        add_flow("Analyzed Screening 1 - Yes", "Predicted Screening 1 - Yes", len(predicted_screening1_yes))
        add_flow("Analyzed Screening 1 - Yes", "Predicted Screening 1 - No", len(predicted_screening1_no))

        # PDF Downloaded
        pdf_downloaded_yes = predicted_screening1_yes[predicted_screening1_yes['PDF_downloaded'] == True]
        pdf_downloaded_no = predicted_screening1_yes[predicted_screening1_yes['PDF_downloaded'] == False]
        add_flow("Predicted Screening 1 - Yes", "PDF Downloaded - Yes", len(pdf_downloaded_yes))
        add_flow("Predicted Screening 1 - Yes", "PDF Downloaded - No", len(pdf_downloaded_no))

        # PDF Embedded
        pdf_embedded_yes = pdf_downloaded_yes[pdf_downloaded_yes['PDF_embedding_error'] == False]
        pdf_embedded_no = pdf_downloaded_yes[pdf_downloaded_yes['PDF_embedding_error'] == True]
        add_flow("PDF Downloaded - Yes", "PDF Embedded - Yes", len(pdf_embedded_yes))
        add_flow("PDF Downloaded - Yes", "PDF Embedded - No", len(pdf_embedded_no))

        # Predicted Screening 2
        predicted_screening2_yes = pdf_embedded_yes[pdf_embedded_yes['predicted_screening2'] == True]
        predicted_screening2_no = pdf_embedded_yes[pdf_embedded_yes['predicted_screening2'] == False]
        add_flow("PDF Embedded - Yes", "Predicted Screening 2 - Yes", len(predicted_screening2_yes))
        add_flow("PDF Embedded - Yes", "Predicted Screening 2 - No", len(predicted_screening2_no))

        # Create the Sankey diagram
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=[f"{label}<br>({label_annotations[label]})" for label in labels],
                color=["green"] + ["blue", "red"] * 5  # Alternating colors for Yes/No
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=["rgba(44, 160, 44, 0.6)" if i % 2 == 0 else "rgba(31, 119, 180, 0.6)" for i in
                       range(len(values))]
            )
        ))

        # Update the layout
        fig.update_layout(title_text="PRISMA Flow Diagram", font_size=10)

        return fig

    @staticmethod
    def _generate_cluster_name(llm, titles):
        """
        Generate a cluster name using a list of titles.

        :param llm: LangChain LLM. The language model to use for generating the cluster name.
        :param titles: list of str. List of titles in the cluster.
        :return: str. Generated name for the cluster.
        """
        prompt_template = "Summarize the following list of article titles into a single line theme below 5 words:\n\n"
        prompt_template += "\n".join(titles)
        prompt_template += "\n\nProvide only the theme without any additional explanation!!!\nTheme:"

        prompt = PromptTemplate.from_template(prompt_template)

        chain = RunnableSequence(prompt | llm)
        response = chain.invoke(input={"prompt": prompt_template})
        theme = response.strip()

        match = re.search(r'"(.*?)"', theme)
        if match:
            theme = match.group(1)

        return theme

    @staticmethod
    def extract_metadata(documents):
        """
        Extract metadata from a list of documents.
        Args:
            documents:

        Returns:
            List of dictionaries containing metadata for each document.
        """
        data = []
        for doc in documents:
            metadata = doc.metadata
            metadata['page_content'] = doc.page_content
            data.append(metadata)
        return data

    @staticmethod
    def get_all_embeddings_and_metadata(faiss_db):
        """
        Get all embeddings and metadata from a FAISS index.
        Args:
            faiss_db:

        Returns:
            Tuple of embeddings and metadata.
        """
        # Access the underlying FAISS index
        index = faiss_db.index
        # Get the total number of vectors
        total_vectors = index.ntotal
        # Retrieve all embeddings
        embeddings = index.reconstruct_n(0, total_vectors)
        # Retrieve metadata
        metadata = [faiss_db.docstore._dict[faiss_db.index_to_docstore_id[i]] for i in range(total_vectors)]
        data = DatasetVisualizer.extract_metadata(metadata)
        metadata = pd.DataFrame(data)

        return embeddings, metadata

# Function to get all embeddings and metadata from a FAISS index

#
# workdir = "/Users/fernando/Documents/Research/academate"
#
# path_name = f"{workdir}/test_old/screening1/vectorDB_screening1.pkl"
# faiss_db = FAISS.load_local(path_name, ollama_embeddings, allow_dangerous_deserialization=True)

#
# def identify_cluster_topics(labels, metadata):
#     rev_per_cluster = 5
#     for i in set(labels):
#         # i = 0
#         print(f"Cluster {i} Theme:", end=" ")
#         contents = "\n".join( metadata[metadata.Cluster == i].sample(rev_per_cluster, random_state=42).page_content.values)
#         response = ollama_llm.invoke(f"What would be the research field of the following articles?:\n:{contents}")
#         print(response)
