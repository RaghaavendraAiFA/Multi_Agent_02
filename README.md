# STAR AI POC

## 🚀 Overview

**STAR AI POC** is an intelligent observability system that combines **Vector Search (RAG)** and **Knowledge Graph (KG)** with **Agentic AI orchestration** to analyze logs, understand infrastructure dependencies, and provide **Root Cause Analysis (RCA)**.

The system ingests logs (e.g., from OpenSearch), transforms them into:

* **Vector Database (Qdrant)** for semantic log retrieval
* **Knowledge Graph (Neo4j)** for infrastructure relationships

It then uses an **AI agent powered by Azure OpenAI** to analyze logs, correlate with system dependencies, and generate actionable insights.

---

## 🧠 Key Features

* 🔍 **Hybrid Retrieval (RAG + KG)**

  * Semantic log search using Qdrant
  * Dependency analysis using Neo4j Knowledge Graph

* 🤖 **Agentic AI System**

  * Intelligent query handling via LangGraph
  * Dynamic use of tools (RAG + KG)

* 🛡️ **Guardrails**

  * Input validation using LLM-based classification
  * Blocks irrelevant or unsafe queries

* 📊 **Root Cause Analysis (RCA)**

  * Combines logs + infrastructure context
  * Provides structured analysis for failures

* 📈 **Observability**

  * Integrated with LangSmith for tracing and monitoring

---


### Flow:

```
User Query
   ↓
FastAPI Backend
   ↓
LangGraph Orchestration
   ↓
Agent (Log Analyzer)
   ↓
 ├── RAG Retriever (Qdrant - Logs)
 ├── KG Retriever (Neo4j - Infra)
   ↓
Azure OpenAI (LLM)
   ↓
RCA Response
```

---

## 🧩 Tech Stack

* **Backend**: FastAPI
* **LLM**: Azure OpenAI (GPT-4o)
* **Vector DB**: Qdrant
* **Knowledge Graph**: Neo4j
* **Agent Framework**: LangGraph
* **Observability**: LangSmith
* **Containerization**: Docker

---

## 📂 Project Structure

```
backend/
├── agentic_ai/
│   ├── agents/
│   ├── tools/
│   └── graphs/
├── api_assist/
│   ├── routes/
│   ├── services/
│   ├── guardrails/
│   ├── llms/
│   └── orchestration/
├── data_ingestion/
├── data/
├── main.py
├── settings.py
```

---

## ⚙️ Setup & Installation

### 1. Clone Repository

```bash
git clone <repo-url>
cd STAR-AI-POC
```

### 2. Create `.env`

Add required credentials:

```
AZURE_CHAT_KEY=
AZURE_CHAT_ENDPOINT=
AZURE_EMBEDDING_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
NEO4J_URI=
NEO4J_USER=
NEO4J_PASSWORD=
```

### 3. Run with Docker

```bash
docker compose build --no-cache
docker compose up -d 

```

### 4. Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## 📡 API Endpoints

### 🔐 Auth

```
POST /auth/login
```

### 📊 Vector DB

```
POST /vdb/insert_vectors
```

### 🧠 Knowledge Graph

```
POST /kg/insert_kg
```

### 🤖 Log Analysis

```
POST /analyze
```

### ❤️ Health Check

```
GET /health
```

---

## 🧪 Example Query

```
"What caused the Redis connection failure in workflow-prod-002?"
```

### Output:

* Relevant logs (from Qdrant)
* Dependency context (from Neo4j)
* AI-generated RCA

---

## 🔮 Future Improvements

* Add **Redis (short-term memory)**
* Add **PostgreSQL (long-term memory)**
* Improve **agent tool selection**
* Add **real-time streaming logs**
* UI dashboard for visualization

---

## 👨‍💻 Author

**Raghu**
AI Engineer | GenAI | RAG | Knowledge Graphs

---

## ⭐ Summary

STAR AI POC demonstrates how to build a **production-style AI system** that combines:

* Retrieval-Augmented Generation (RAG)
* Knowledge Graph reasoning
* Agentic workflows

👉 Delivering intelligent, context-aware **log analysis and RCA**.
