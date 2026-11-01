import os
import glob
from typing import List, Any
from typing_extensions import TypedDict

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END

# =========================================================
# 1. INIT
# =========================================================

load_dotenv(override=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# =========================================================
# 2. LOAD PDFs
# =========================================================

all_docs = []
pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))

print(f"{len(pdf_files)} PDF trouvés")

for pdf_path in pdf_files:

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    for d in docs:
        d.metadata["source_file"] = os.path.basename(pdf_path)
        d.metadata["full_path"] = pdf_path

    all_docs.extend(docs)

print(f"{len(all_docs)} pages chargées")

# =========================================================
# 3. SPLIT
# =========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(all_docs)

# =========================================================
# 4. EMBEDDINGS
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================================================
# 5. VECTORSTORE
# =========================================================

if os.path.exists(CHROMA_DIR):
    import shutil
    shutil.rmtree(CHROMA_DIR)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
    collection_name="invoices"
)

vectorstore.persist()

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# =========================================================
# 6. LLM
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# =========================================================
# 7. STATE
# =========================================================

class AgentState(TypedDict):
    query: str
    original_query: str
    thought: str
    documents: List[Any]
    context: str
    generation: str
    is_relevant: str
    retry_count: int
    messages: List[Any]

# =========================================================
# 8. HELPER
# =========================================================

def merge(state, update):
    return {**state, **update}

# =========================================================
# 9. Init node
# =========================================================

def init_node(state: AgentState):
    return merge(state, {
        "original_query": state.get("original_query", state["query"]),
        "documents": state.get("documents", []),
        "generation": state.get("generation", ""),
        "is_relevant": state.get("is_relevant", ""),
        "retry_count": state.get("retry_count", 0),
        "thought": state.get("thought", ""),
        "context": state.get("context", ""),
        "messages": state.get("messages", [])
    })
    

# =========================================================
# 10. 🧠 THOUGHT NODE (ReAct Reasoning)
# =========================================================

def thought_node(state: AgentState):

    prompt = f"""
Tu es un agent RAG spécialisé dans les factures.

Analyse la question et fais 2 choses :

1. Reformule une requête optimisée pour rechercher dans des factures
2. Explique brièvement ce que tu cherches

QUESTION:
{state["query"]}

Réponds EXACTEMENT au format :

QUERY: ...
THOUGHT: ...
"""

    response = llm.invoke(prompt).content

    lines = response.split("\n")

    new_query = state["query"]
    thought = response

    for line in lines:
        if line.startswith("QUERY:"):
            new_query = line.replace("QUERY:", "").strip()

    return merge(state, {
        "query": new_query,
        "thought": thought
    })

# =========================================================
# 11. RETRIEVE
# =========================================================

def retrieve_node(state: AgentState):

    docs = retriever.invoke(state["query"])

    context = []

    for d in docs:
        context.append(
            f"""
FACTURE: {d.metadata.get("source_file")}
CHEMIN: {d.metadata.get("full_path")}

{d.page_content}
"""
        )

    return merge(state, {
        "documents": docs,
        "context": "\n\n".join(context)
    })

# =========================================================
# 12. GRADE
# =========================================================

def grade_node(state: AgentState):

    if not state["documents"]:
        return merge(state, {"is_relevant": "no"})

    original_query = state.get("original_query", state.get("query", ""))

    prompt = f"""
Réponds yes ou no uniquement.

QUESTION: {original_query}

CONTEXTE:
{state["context"]}
"""

    res = llm.invoke(prompt).content.lower()

    decision = "yes" if "yes" in res else "no"

    return merge(state, {"is_relevant": decision})

# =========================================================
# 13. REWRITE
# =========================================================

def rewrite_node(state: AgentState):

    prompt = f"""
Améliore cette requête pour une recherche dans des factures :

{state["query"]}
"""

    new_q = llm.invoke(prompt).content.strip()

    return merge(state, {
        "query": new_q,
        "retry_count": state["retry_count"] + 1
    })

# =========================================================
# 14. GENERATE
# =========================================================

def generate_node(state: AgentState):

    prompt = f"""
Tu es un assistant de lecture de factures.

Réponds uniquement avec les factures fournies.

CONTEXTE:
{state["context"]}

QUESTION:
{state["original_query"]}
"""

    res = llm.invoke(prompt).content

    return merge(state, {"generation": res})

# =========================================================
# 15. FALLBACK
# =========================================================

def fallback_node(state: AgentState):

    return merge(state, {
        "generation": "Aucune information trouvée dans les factures."
    })

# =========================================================
# 16. ROUTING
# =========================================================

def route(state):

    if state["is_relevant"] == "yes":
        return "generate"

    if state["retry_count"] < 2:
        return "rewrite"

    return "fallback"

# =========================================================
# 17. GRAPH
# =========================================================

workflow = StateGraph(AgentState)

workflow.add_node("init", init_node)
workflow.add_node("thought", thought_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("generate", generate_node)
workflow.add_node("fallback", fallback_node)

workflow.set_entry_point("init")

workflow.add_edge("init", "thought")
workflow.add_edge("thought", "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_edge("rewrite", "thought")

workflow.add_conditional_edges(
    "grade",
    route,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "fallback": "fallback"
    }
)

workflow.add_edge("generate", END)
workflow.add_edge("fallback", END)

app = workflow.compile()

# =========================================================
# 18. RUN
# =========================================================

def run(question: str):

    result = app.invoke({
        "query": question,
        "original_query": question,
        "thought": "",
        "documents": [],
        "context": "",
        "generation": "",
        "is_relevant": "",
        "retry_count": 0,
        "messages": []
    })

    print("\n==============================")
    print("QUESTION:", question)
    print("==============================\n")

    print(result["generation"])

    print("\n--- THOUGHT ---")
    print(result.get("thought", ""))

# =========================================================
# 19. TEST
# =========================================================

if __name__ == "__main__":

    while True:

        q = input("\nQuestion (q pour quitter): ")

        if q.lower() == "q":
            break

        run(q)