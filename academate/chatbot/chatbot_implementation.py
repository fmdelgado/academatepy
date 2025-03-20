# academate/chatbot/chatbot_implementation.py

import logging
import re
from typing import Any, Dict, List, Optional

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logger = logging.getLogger(__name__)


class AcademateChat:
    """
    A robust chatbot implementation for Academate that allows users to
    query the screened research articles.
    """

    def __init__(self, llm, embeddings, verbose=False):
        """
        Initialize the AcademateChat with language model and embeddings.

        Args:
            llm: Language model for generating responses
            embeddings: Embeddings for semantic search
            verbose: Whether to show detailed logs
        """
        self.llm = llm
        self.embeddings = embeddings
        self.verbose = verbose
        self.vectorstore = None
        self.retriever = None

        # Set up logging level
        logging_level = logging.DEBUG if verbose else logging.INFO
        logger.setLevel(logging_level)

        # Initialize the RAG prompt template
        self.rag_prompt = PromptTemplate(
            template="""You are a research assistant helping with a systematic literature review. 
Answer the question based ONLY on the provided context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Only use information from the provided context
2. Include citations after each statement in the format [Article X]
3. If the context doesn't contain the information, say "I cannot answer this question based on the available articles"
4. Be concise but thorough

ANSWER:
""",
            input_variables=["context", "question"],
        )

        # Create the RAG chain
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()

    def create_document_store(self, articles_df, content_column="Record", persist_dir=None):
        """
        Create a document store from a DataFrame of articles.

        Args:
            articles_df: DataFrame containing screened articles
            content_column: Column name containing the article text
            persist_dir: Directory to persist the vector store

        Returns:
            self: For method chaining
        """
        # Filter articles that passed screening
        if 'predicted_screening2' in articles_df.columns and articles_df['predicted_screening2'].any():
            selected_df = articles_df[articles_df['predicted_screening2'] == True].copy()
            screening_type = "screening2"
        elif 'predicted_screening1' in articles_df.columns and articles_df['predicted_screening1'].any():
            selected_df = articles_df[articles_df['predicted_screening1'] == True].copy()
            screening_type = "screening1"
        else:
            # If no screening columns, use all articles
            selected_df = articles_df.copy()
            screening_type = "all"

        logger.info(f"Creating document store with {len(selected_df)} articles that passed {screening_type}")

        # Create Document objects
        documents = []
        for idx, row in selected_df.iterrows():
            content = row[content_column]
            if not isinstance(content, str) or not content.strip():
                continue

            metadata = {
                'uniqueid': str(row.get('uniqueid', idx)),
                'title': str(row.get('title', f"Article {idx}")),
                'screening_type': screening_type,
                'source': f"Article {row.get('uniqueid', idx)}"
            }

            # Add any additional available metadata
            for col in ['authors', 'year', 'journal', 'doi']:
                if col in row and not pd.isna(row[col]):
                    metadata[col] = str(row[col])

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        # Split documents into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")

        # Create vector store
        if persist_dir:
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=persist_dir
            )
            # Persist the vector store
            logger.info(f"Persisted vector store to {persist_dir}")
        else:
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings
            )

        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Use Maximum Marginal Relevance
            search_kwargs={"k": 5, "fetch_k": 20}  # Fetch 20, return 5 diverse results
        )

        return self

    def load_document_store(self, persist_dir):
        """
        Load an existing document store from disk.

        Args:
            persist_dir: Directory where the vector store is persisted

        Returns:
            self: For method chaining
        """
        logger.info(f"Loading vector store from {persist_dir}")
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )

        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )

        return self

    def query(self, question):
        """
        Query the chatbot with a question.

        Args:
            question: Question to ask about the articles

        Returns:
            dict: Result containing the answer and source documents
        """
        if not self.retriever:
            error_message = "Document store not initialized. Call create_document_store() or load_document_store() first."
            logger.error(error_message)
            return {
                "answer": error_message,
                "documents": [],
                "success": False
            }

        try:
            # Retrieve relevant documents
            docs = self.retriever.invoke(question)

            if not docs:
                return {
                    "answer": "I cannot answer this question based on the available articles.",
                    "documents": [],
                    "success": True
                }

            # Format context with citations
            context_parts = []
            for i, doc in enumerate(docs):
                article_id = doc.metadata.get('uniqueid', f"Article {i + 1}")
                context_parts.append(f"[Article {i + 1}] {doc.page_content}")

            context = "\n\n".join(context_parts)

            # Create article reference section
            references = []
            for i, doc in enumerate(docs):
                title = doc.metadata.get('title', 'Untitled')
                authors = doc.metadata.get('authors', '')
                year = doc.metadata.get('year', '')

                reference = f"[Article {i + 1}]: {title}"
                if authors and year:
                    reference += f" ({authors}, {year})"
                elif year:
                    reference += f" ({year})"

                references.append(reference)

            context += "\n\nArticle References:\n" + "\n".join(references)

            # Generate answer
            answer = self.rag_chain.invoke({
                "context": context,
                "question": question
            })

            # Extract citation references
            citation_pattern = r'\[Article\s*(\d+)\]'
            citations = list(set(re.findall(citation_pattern, answer)))

            # Create source mapping
            source_mapping = {}
            for citation in citations:
                idx = int(citation) - 1
                if 0 <= idx < len(docs):
                    source_mapping[f"Article {citation}"] = docs[idx]

            return {
                "answer": answer,
                "documents": docs,
                "sources": source_mapping,
                "success": True
            }

        except Exception as e:
            error_message = f"Error processing question: {str(e)}"
            logger.error(error_message)
            import traceback
            logger.debug(traceback.format_exc())

            return {
                "answer": "I encountered an error while trying to answer your question.",
                "error": error_message,
                "success": False
            }

    def get_document_count(self):
        """Get the number of documents in the vector store"""
        if self.vectorstore:
            return self.vectorstore._collection.count()
        return 0