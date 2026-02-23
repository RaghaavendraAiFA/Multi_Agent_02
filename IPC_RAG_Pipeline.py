
#  ipc_rag_pipeline.py 

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from config import *

# Determine local vector DB directory (fallback to "Indianpenalcode" if not set)
INDEX_DIR = Path(VECTOR_DB_PATH) if "VECTOR_DB_PATH" in globals() and VECTOR_DB_PATH else Path("Indianpenalcode")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# EMBEDDINGS (needed for load or create)
embeddings = AzureOpenAIEmbeddings(
    model=AZURE_OPENAI_EMB_MODEL_NAME,
    azure_endpoint=AZURE_OPENAI_EMB_ENDPOINT,
    api_key=AZURE_OPENAI_EMB_API_KEY,
    azure_deployment=AZURE_OPENAI_EMB_DEPLOYMENT_NAME,
    openai_api_version=AZURE_OPENAI_EMB_API_VERSION,
)

#  STEP 2 — CREATE FAISS INDEX IF MISSING

def build_or_load_vectorstore():
    # consider index exists if directory contains any files
    if any(INDEX_DIR.iterdir()):
        print(f"Found existing index at '{INDEX_DIR}', loading...")
        try:
            store = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
            return store
        except Exception as e:
            print("Failed to load existing index, will rebuild. Error:", e)

    # Build index
    print("No valid index found — creating FAISS index from PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"length of chunks: {len(chunks)}")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    try:
        vectorstore.save_local(str(INDEX_DIR))
        print(f"Saved FAISS index to '{INDEX_DIR}'")
    except Exception as e:
        print("Warning: could not save FAISS index:", e)

    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)


local_store = build_or_load_vectorstore()
retriever = local_store.as_retriever(search_kwargs={"k": 4})

# LLM

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_CM_ENDPOINT,
    api_key=AZURE_OPENAI_CM_API_KEY,
    azure_deployment=AZURE_OPENAI_CM_DEPLOYMENT_NAME,
    openai_api_version=AZURE_OPENAI_CM_API_VERSION,
    temperature=TEMPERATURE,
)

# MEMORY

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)

# PROMPT

system_prompt = """You are an expert Indian Penal Code (IPC) Legal Agent.

Rules:
1. If the question is a greeting, reply politely.
2. Answer ONLY using the provided IPC document context OR previous chat history for follow-up questions.
3. Always mention the exact IPC Section number.
4. Include: offense description, punishment, and bailable/non-bailable status.
5. If the answer is not in the context, reply exactly:
   "This information is not available in the provided IPC document."
6. Never guess or use outside knowledge.
7. End every answer with: "Note: Please consult a qualified lawyer for legal advice."
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", """
Context from IPC Document:
{context}

Conversation so far:
{chat_history}

Question: {question}
""")
])

# ─────────────────────────────────────────────────────
#  STEP 7 — QA CHAIN
# ─────────────────────────────────────────────────────
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=False,
    verbose=False,
)

print("IPC QA Chain is ready!\n")

def get_ipc_qa_chain():
    return qa_chain

if __name__ == "__main__":
    result = qa_chain.invoke({"question": "exactly as it is in pdf Section 23. Wrongful gain complete information give me"})
    print(result["answer"])