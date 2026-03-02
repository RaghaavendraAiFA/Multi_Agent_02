
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from config import * 

#folder paths 
UPLOAD_DIR       = Path("uploaded_files")        # uploaded PDFs saved here
UPLOAD_INDEX_DIR = Path("uploaded_faiss_index")  # FAISS index saved here

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_INDEX_DIR.mkdir(parents=True, exist_ok=True)
#  GLOBAL STATE
_upload_state = {
    "qa_chain"   : None,   # active qa_chain (None = no upload yet)
    "filename"   : None,   # uploaded filename
    "total_pages": 0,      # number of pages in PDF
    "total_chunks": 0,     # number of chunks created
}

#  AZURE EMBEDDINGS

embeddings = AzureOpenAIEmbeddings(
    model             = AZURE_OPENAI_EMB_DEPLOYMENT_NAME,
    azure_endpoint    = AZURE_OPENAI_EMB_ENDPOINT,
    api_key           = AZURE_OPENAI_EMB_API_KEY,
    azure_deployment  = AZURE_OPENAI_EMB_DEPLOYMENT_NAME,
    openai_api_version= AZURE_OPENAI_EMB_API_VERSION,
)

#  LLM
llm = AzureChatOpenAI(
    azure_endpoint    = AZURE_OPENAI_CM_ENDPOINT,
    api_key           = AZURE_OPENAI_CM_API_KEY,
    azure_deployment  = AZURE_OPENAI_CM_DEPLOYMENT_NAME,
    openai_api_version= AZURE_OPENAI_CM_API_VERSION,
    temperature       = TEMPERATURE,
)
#  STEP 1 — LOAD PDF
def load_pdf(pdf_path: str) -> list:

    print(f"  [Pipeline] Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs   = loader.load()
    print(f"  [Pipeline] Pages loaded: {len(docs)}")
    return docs

#  STEP 2 — CHUNK
def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  [Pipeline] Chunks created: {len(chunks)}")
    return chunks

#  STEP 3 — BUILD FAISS
def build_and_save_faiss(chunks: list) -> FAISS:
    """Embed chunks using Azure OpenAI and save FAISS index to disk."""

    # Clear old FAISS index before building new one
    if UPLOAD_INDEX_DIR.exists():
        shutil.rmtree(UPLOAD_INDEX_DIR)
    UPLOAD_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  [Pipeline] Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(UPLOAD_INDEX_DIR))
    print(f"  [Pipeline] FAISS index saved to: {UPLOAD_INDEX_DIR}")
    return vectorstore

#  STEP 4 — BUILD QA CHAIN
def build_qa_chain(vectorstore: FAISS, filename: str) -> ConversationalRetrievalChain:
    """Build the full RAG QA chain from vectorstore."""

    # Retriever — fetches top 5 relevant chunks per query
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Memory — remembers conversation history for follow-up questions
    memory = ConversationBufferMemory(
        memory_key  = "chat_history",
        return_messages = True,
        output_key  = "answer",
    )

    # Prompt — strict document-only answering
    system_prompt = f"""You are an expert IPC Legal Document Agent.
You are answering questions from the uploaded document: "{filename}"

STRICT RULES:
1. Answer ONLY from the provided document context.
2. Always cite exact section numbers, clause numbers when available.
3. Include: what the section says, punishment, bailable/non-bailable status.
4. For follow-up questions, use the chat history to maintain context.
5. If the answer is NOT in the document, say exactly:
   "This information is not found in the uploaded document."
6. Never use outside knowledge or guess.
7. End every answer with: "Note: Consult a qualified lawyer for legal advice."
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nHistory:\n{chat_history}\n\nQuestion: {question}"),
    ])

    # Build QA chain
    chain = ConversationalRetrievalChain.from_llm(
        llm                    = llm,
        retriever              = retriever,
        memory                 = memory,
        combine_docs_chain_kwargs = {"prompt": prompt},
        return_source_documents= False,
        verbose                = False,
    )

    print(f"  [Pipeline] QA chain built for: {filename}")
    return chain


#  MAIN FUNCTION — process_uploaded_pdf

def process_uploaded_pdf(pdf_path: str, filename: str) -> dict:

    global _upload_state

    print(f"\n[IPC Pipeline] Starting pipeline for: {filename}")

    # Run all steps
    docs        = load_pdf(pdf_path)
    chunks      = chunk_documents(docs)
    vectorstore = build_and_save_faiss(chunks)
    qa_chain    = build_qa_chain(vectorstore, filename)

    # Store in global state
    _upload_state["qa_chain"]    = qa_chain
    _upload_state["filename"]    = filename
    _upload_state["total_pages"] = len(docs)
    _upload_state["total_chunks"]= len(chunks)

    print(f"[IPC Pipeline] ✅ Pipeline complete for: {filename}")

    return {
        "filename"    : filename,
        "total_pages" : len(docs),
        "total_chunks": len(chunks),
        "status"      : "ready",
    }

#  GETTER FUNCTIONS — used by IPC Agent and FastAPI
def get_current_qa_chain():
    return _upload_state["qa_chain"]

def get_upload_status() -> dict:
    if _upload_state["qa_chain"] is None:
        return {
            "uploaded"    : False,
            "filename"    : None,
            "total_pages" : 0,
            "total_chunks": 0,
            "message"     : "No document uploaded yet. Please upload a PDF first.",
        }
    return {
        "uploaded"    : True,
        "filename"    : _upload_state["filename"],
        "total_pages" : _upload_state["total_pages"],
        "total_chunks": _upload_state["total_chunks"],
        "message"     : f"Active document: {_upload_state['filename']}",
    }


def clear_upload():
    global _upload_state
    _upload_state = {
        "qa_chain"   : None,
        "filename"   : None,
        "total_pages": 0,
        "total_chunks": 0,
    }
    # Clean up saved files
    if UPLOAD_INDEX_DIR.exists():
        shutil.rmtree(UPLOAD_INDEX_DIR)
        UPLOAD_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print("[IPC Pipeline] Upload cleared.")