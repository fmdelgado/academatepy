import json
import re
import logging
from typing import List, Any
from typing_extensions import TypedDict

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import END, StateGraph, START

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

## Helper functions

def format_docs(docs):
    """Helper to join doc content if needed by grader."""
    return "\n\n".join(doc.page_content for doc in docs)

def clean_answer(answer):
    """
    Cleans up the answer string to extract key-value pairs and format them nicely.
    """
    if "I'm sorry" in answer or "do not contain information" in answer:
        return answer

    pattern = r'(\w+\s?(?:\(\w+\s?\w+\.\s?\d+\))?|\w+)\s*:\s*([^,]+|.+?)(?=,\s*\w+\s*:|\Z)'
    matches = re.findall(pattern, answer)
    if not matches:
        return answer

    formatted_answer = ""
    for key, value in matches:
        formatted_answer += f"- **{key.strip()}**: {value.strip()}\n"
    return formatted_answer

def map_citations_to_sources(generation, documents):
    """Match [Source X] citations in generation to actual document chunks."""
    generation_text = generation.strip()
    citation_pattern = r'\[Source\s*(\d+)\]'
    citations = re.findall(citation_pattern, generation_text)
    logger.info(f"---CITATIONS FOUND: {citations}---")
    unique_citations = set(citations)
    source_mapping = {}
    for citation in unique_citations:
        idx = int(citation) - 1  # zero-based
        if 0 <= idx < len(documents):
            source_mapping[f"Source {citation}"] = documents[idx]
        else:
            logger.warning(f"Invalid citation [Source {citation}]")
    return source_mapping

def remap_source_citations(generated_text, documents):
    """
    Re-map model-produced [Source X] citations to the correct doc index
    based on the final doc list order in `documents`.
    """
    official_map = {}
    for i, doc in enumerate(documents, start=1):
        official_map[f"Source {i}"] = doc

    pattern = r"\[Source\s+(\d+)\]"
    replacements = []
    for match in re.findall(pattern, generated_text):
        old_str = f"[Source {match}]"
        if old_str in official_map:
            doc_obj = official_map[old_str]
            true_index = documents.index(doc_obj) + 1
            new_str = f"[Source {true_index}]"
        else:
            new_str = ""
        replacements.append((old_str, new_str))

    remapped_text = generated_text
    for old, new in replacements:
        remapped_text = remapped_text.replace(old, new)
    return remapped_text

# -------------------------------------------
#  Prompt Setup
# -------------------------------------------
# This prompt template was previously named "prompt" but to avoid confusion
# with user questions we define a dedicated retrieval prompt.
retrieval_prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question.
Here is the retrieved document:

{document}

Here is the user question: {question}
Provide a JSON response with a key "score" that is "yes" if relevant and "no" otherwise.""",
    input_variables=["question", "document"],
)

rag_prompt = PromptTemplate(
    template="""You are an assistant that answers questions based on the provided context.

Context:
{context}

Question:
{question}

**It is crucial that you only answer based on the provided context.** 
Provide a concise answer to the question based solely on the context above. 
**After every sentence or factual statement, include a citation in the format [Source X].** 
Use the Source IDs provided in the context and sources list. 
+ Ignore any bracketed references ([5], [6], etc.) that appear inside the document text itself; 
+ only use the [Source #] labels assigned to each chunk in the Sources section below.

**If you cannot answer the question based on the context, respond with:** 
"I'm sorry, but the documents do not contain information about your question."

Answer:
""",
    input_variables=["context", "question"],
)

hallucination_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. 
Here are the facts:
------- 
{documents} 
------- 
Here is the answer: {generation}
Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. 
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "documents"],
)

answer_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. 
Here is the answer:
------- 
{generation} 
------- 
Here is the question: {question}
Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. 
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],
)

re_write_prompt = PromptTemplate(
    template="""You are a question re-writer that converts an input question to a better version that is optimized 
for vectorstore retrieval. Look at the initial question and formulate an improved question. 
Here is the initial question:

{question}

Output only the improved question with no preamble or explanation.""",
    input_variables=["question"],
)

# -------------------------------------------
#  StateGraph Nodes
# -------------------------------------------
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Any]
    retriever: Any

def retrieve(state):
    logger.info("---RETRIEVE---")
    question = state["question"]
    retriever = state["retriever"]
    documents = retriever.invoke(question)
    state["documents"] = documents
    return state

def grade_documents(state):
    logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    for doc in documents:
        try:
            response = state["retrieval_grader"].invoke(
                {"question": question, "document": doc.page_content}
            )
            grade = None
            if isinstance(response, dict) and "score" in response:
                grade = response["score"]
            else:
                json_match = re.search(r"\{.*?}", str(response))
                if json_match:
                    try:
                        response_json = json.loads(json_match.group(0))
                        grade = response_json.get("score")
                    except json.JSONDecodeError:
                        grade = None
                else:
                    resp_lower = str(response).lower()
                    if "yes" in resp_lower:
                        grade = "yes"
                    elif "no" in resp_lower:
                        grade = "no"

            if grade and grade.lower() == "yes":
                logger.info(f"---GRADE: DOCUMENT RELEVANT--- Source={doc.metadata.get('source', 'N/A')}")
                filtered_docs.append(doc)
            else:
                logger.info(f"---GRADE: DOCUMENT NOT RELEVANT--- Source={doc.metadata.get('source', 'N/A')}")
        except Exception as e:
            logger.error(f"---ERROR: Error grading document: {e}")
    state["documents"] = filtered_docs
    return state

def decide_to_generate(state):
    logger.info("---ASSESS GRADED DOCUMENTS---")
    if not state["documents"]:
        logger.info("---DECISION: NO DOCS => handle_no_answer---")
        return "handle_no_answer"
    else:
        logger.info("---DECISION: proceed to generate---")
        return "generate"

def generate(state):
    logger.info("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    if not documents:
        generation = "I'm sorry, but the documents do not contain information about your question."
        state["generation"] = generation
        return state

    context = "\n\n".join(
        f"[Source {i + 1}] {doc.page_content}" for i, doc in enumerate(documents)
    )
    sources_str = "\n".join(
        f"[Source {i + 1}]: {doc.metadata.get('source', 'N/A')} ..." for i, doc in enumerate(documents)
    )
    full_context = f"{context}\n\nSources:\n{sources_str}"
    generation = state["rag_chain"].invoke({"context": full_context, "question": question})
    logger.info(f"---RAW GENERATION: {generation}---")
    state["generation"] = generation
    return state

def grade_generation_v_documents_and_question(state):
    logger.info("---CHECK HALLUCINATIONS---")
    question = state["question"]
    generation = state["generation"]
    documents = state["documents"]
    source_mapping = map_citations_to_sources(generation, documents)
    state["source_mapping"] = source_mapping
    mapped_docs = list(source_mapping.values())
    hall_score = state["hallucination_grader"].invoke(
        {"documents": format_docs(mapped_docs), "generation": generation}
    )
    logger.info(f"---HALLUCINATION CHECK SCORE: {hall_score}---")
    hall_grade = hall_score.get("score", None)
    if hall_grade == "yes":
        ans_score = state["answer_grader"].invoke({"question": question, "generation": generation})
        logger.info(f"---ANSWER CHECK SCORE: {ans_score}---")
        ans_grade = ans_score.get("score", None)
        if ans_grade == "yes":
            logger.info("---DECISION: generation addresses question -> END---")
            state["documents"] = mapped_docs
            return "useful"
        else:
            logger.info("---DECISION: does not address question -> no_answer---")
            return "no_answer"
    else:
        logger.info("---DECISION: generation is not grounded -> no_answer---")
        return "no_answer"

def handle_no_answer(state):
    logger.info("---HANDLE NO ANSWER---")
    generation = "I'm sorry, but the documents do not contain information about your question."
    state["generation"] = generation
    return state

# -------------------------------------------
#  ChatbotApp Class Definition
# -------------------------------------------
class ChatbotApp:
    def __init__(self, llm):
        """
        Initialize the chatbot workflow with the given LLM.
        """
        self.llm = llm

        # Build the chains using self.llm.
        # Note: We now use 'retrieval_prompt' (a PromptTemplate) instead of 'prompt'
        # to avoid accidentally composing a plain string.
        self.retrieval_grader = retrieval_prompt | self.llm | JsonOutputParser()
        self.rag_chain = rag_prompt | self.llm | StrOutputParser()
        self.hallucination_grader = hallucination_prompt | self.llm | JsonOutputParser()
        self.answer_grader = answer_grader_prompt | self.llm | JsonOutputParser()
        self.question_rewriter = re_write_prompt | self.llm | StrOutputParser()

        # Build the workflow
        self.workflow = StateGraph(GraphState)
        self.workflow.add_node("retrieve", retrieve)
        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_node("grade_documents", grade_documents)
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_node("generate", generate)
        self.workflow.add_node("handle_no_answer", handle_no_answer)
        self.workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {"handle_no_answer": "handle_no_answer", "generate": "generate"},
        )
        self.workflow.add_edge("handle_no_answer", END)
        self.workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {"useful": END, "no_answer": "handle_no_answer"},
        )

        # Compile the workflow
        self.app = self.workflow.compile()

    def invoke(self, inputs):
        """
        Invoke the compiled workflow. Inject prompt chains into the state.
        """
        inputs["retrieval_grader"] = self.retrieval_grader
        inputs["rag_chain"] = self.rag_chain
        inputs["hallucination_grader"] = self.hallucination_grader
        inputs["answer_grader"] = self.answer_grader
        inputs["question_rewriter"] = self.question_rewriter
        inputs.setdefault("llm", self.llm)
        return self.app.invoke(inputs)

# For backwards compatibility, expose clean_answer and remap_source_citations.
__all__ = ["ChatbotApp", "clean_answer", "remap_source_citations"]