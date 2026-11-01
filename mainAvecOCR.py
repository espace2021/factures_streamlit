import os
import glob
from typing import List, Any
from typing_extensions import TypedDict

from dotenv import load_dotenv


# easyocr est une bibliothèque Python d'OCR (Optical Character Recognition) basée sur du deep learning.
# Elle prend une image en entrée et détecte + reconnaît le texte 
import easyocr

# Initialisation unique du reader (télécharge les modèles au premier lancement)
# Uvicorn avec reload=False importe quand même le module deux fois (au chargement + au démarrage). Le ocr_reader est donc créé deux fois, d'où le double Using CPU.
# Protéger l'initialisation avec un guard
if "ocr_reader" not in globals():
    ocr_reader = easyocr.Reader(["fr", "en"], gpu=False)

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.runnables.config import RunnableConfig

from skills.skill_loader import load_skill

# =========================================================
# 1. INIT
# =========================================================

load_dotenv(override=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")      # ← dossier PNG
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

SKILL_PATH = os.path.join(BASE_DIR, "skills", "invoice_skill.md")
INVOICE_SKILL = load_skill(SKILL_PATH)

# =========================================================
# 2. OCR HELPERS
# =========================================================

def ocr_image(image_path: str) -> str:
    results = ocr_reader.readtext(image_path, detail=0, paragraph=True)
    return "\n".join(results)


# =========================================================
# 3. CUSTOM LOADER (compatible LangChain)
# =========================================================

class ImageInvoiceLoader(BaseLoader):
    """
    Loader LangChain pour des factures en PNG.
    Chaque image → 1 Document avec métadonnées source_file + full_path.
    """

    def __init__(self, image_paths: List[str], lang: str = "fra+eng"):
        self.image_paths = image_paths
        self.lang = lang

    def lazy_load(self):
        for path in self.image_paths:
            try:
                text = ocr_image(path)
                if not text:
                    print(f"⚠️  Aucun texte extrait de {path}")
                    continue
                yield Document(
                    page_content=text,
                    metadata={
                        "source_file": os.path.basename(path),
                        "full_path": path,
                    }
                )
            except Exception as e:
                print(f"❌ Erreur OCR sur {path} : {e}")

    def load(self) -> List[Document]:
        return list(self.lazy_load())


# =========================================================
# 4. LOAD IMAGES via OCR
# =========================================================

image_files = glob.glob(os.path.join(IMAGE_DIR, "*.png"))
# Inclure aussi les .jpg/.jpeg si besoin :
# image_files += glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
# image_files += glob.glob(os.path.join(IMAGE_DIR, "*.jpeg"))

print(f"{len(image_files)} image(s) PNG trouvée(s)")

loader = ImageInvoiceLoader(image_paths=image_files)
all_docs = loader.load()

print(f"{len(all_docs)} document(s) chargé(s) après OCR")

# Après loader.load()
for doc in all_docs:
    print(f"\n📄 {doc.metadata['source_file']}")
    print(doc.page_content[:500])  # affiche les 500 premiers caractères

# =========================================================
# 5. SPLIT
# =========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(all_docs)
print(f"{len(chunks)} chunk(s) créé(s)")

# =========================================================
# 6. EMBEDDINGS
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================================================
# 7. VECTORSTORE
# =========================================================

if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    print("🔄 Chargement de la base ChromaDB existante...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="invoices"
    )
else:
    print("✨ Création d'une nouvelle base ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="invoices"
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# =========================================================
# 8. LLM
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# =========================================================
# 9. STATE
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
# 10. HELPER
# =========================================================

def merge(state, update):
    return {**state, **update}

# =========================================================
# 11. NODES (identiques à votre version originale)
# =========================================================

def init_node(state: AgentState):
    return merge(state, {
        "original_query": state.get("original_query", state["query"]),
        "documents":      state.get("documents", []),
        "generation":     state.get("generation", ""),
        "is_relevant":    state.get("is_relevant", ""),
        "retry_count":    state.get("retry_count", 0),
        "thought":        state.get("thought", ""),
        "context":        state.get("context", ""),
        "messages":       state.get("messages", [])
    })


def thought_node(state: AgentState):
    prompt = f"""
Tu es un agent RAG spécialisé dans les factures.

Reformule la question en une requête courte et précise (max 10 mots)
pour rechercher dans des factures.

QUESTION: {state["query"]}

Réponds EXACTEMENT au format :
QUERY: [requête courte]
THOUGHT: [ce que tu cherches en 1 phrase]
"""
    response = llm.invoke(prompt).content
    lines = response.split("\n")
    new_query = state["query"]
    for line in lines:
        if line.startswith("QUERY:"):
            new_query = line.replace("QUERY:", "").strip()
    return merge(state, {"query": new_query, "thought": response})


def retrieve_node(state: AgentState):
    docs = retriever.invoke(state["query"])
    context = []
    for d in docs:
        context.append(
            f"""
FACTURE: {d.metadata.get("source_file")}
CHEMIN:  {d.metadata.get("full_path")}

{d.page_content}
"""
        )
    return merge(state, {"documents": docs, "context": "\n\n".join(context)})


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


def rewrite_node(state: AgentState):
    prompt = f"""
Améliore cette requête pour une recherche dans des factures :

{state["query"]}
"""
    new_q = llm.invoke(prompt).content.strip()
    return merge(state, {"query": new_q, "retry_count": state["retry_count"] + 1})


def generate_node(state: AgentState):
    prompt = f"""
{INVOICE_SKILL}

---

CONTEXTE:
{state["context"]}

QUESTION:
{state.get("original_query", state.get("query", ""))}
"""
    res = llm.invoke(prompt).content
    return merge(state, {"generation": res})


def fallback_node(state: AgentState):
    return merge(state, {"generation": "Aucune information trouvée dans les factures."})


def route(state):
    if state["is_relevant"] == "yes":
        return "generate"
    if state["retry_count"] < 2:
        return "rewrite"
    return "fallback"

# =========================================================
# 12. GRAPH
# =========================================================

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("init",     init_node)
    workflow.add_node("thought",  thought_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade",    grade_node)
    workflow.add_node("rewrite",  rewrite_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("fallback", fallback_node)

    workflow.set_entry_point("init")
    workflow.add_edge("init",     "thought")
    workflow.add_edge("thought",  "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("rewrite",  "thought")

    workflow.add_conditional_edges(
        "grade", route,
        {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"}
    )

    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()

# =========================================================
# 13. FASTAPI + LIFESPAN
# =========================================================

app_state = {}
app = build_graph()

@asynccontextmanager
async def lifespan(api: FastAPI):
    app_state["graph"]     = app
    app_state["llm"]       = llm
    app_state["retriever"] = retriever
    app_state["skill"]     = INVOICE_SKILL
    print("✅ Graph + LLM + Retriever + Skill initialisés")
    yield
    app_state.clear()
    print("🧹 App state cleared")

api = FastAPI(title="Invoice RAG API (OCR)", lifespan=lifespan)

# =========================================================
# 14. ENDPOINTS
# =========================================================

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question: str
    answer: str
    thought: str | None = None


async def invoice_graph(runtime=None, config: RunnableConfig | None = None):
    if not app_state:
        raise RuntimeError("Application non initialisée.")
    graph = app_state["graph"]
    query = (config or {}).get("configurable", {}).get("query", "")
    result = await graph.ainvoke({
        "query":          query,
        "original_query": query,
        "thought":        "",
        "documents":      [],
        "context":        "",
        "generation":     "",
        "is_relevant":    "",
        "retry_count":    0,
        "messages":       []
    }, config=config or RunnableConfig(tags=["api"]))
    return {"answer": result.get("generation", ""), "thought": result.get("thought", "")}


@api.post("/ask")
async def ask(body: AskRequest):
    config = RunnableConfig(tags=["api"], configurable={"query": body.question})
    result = await invoice_graph(config=config)
    return AskResponse(question=body.question, thought=result.get("thought"), answer=result.get("answer"))


# =========================================================
# 15. MAIN
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:api", host="0.0.0.0", port=8000, reload=False)