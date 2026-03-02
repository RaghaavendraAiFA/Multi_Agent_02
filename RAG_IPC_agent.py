from typing import Annotated, TypedDict
import operator
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END

from IPC_RAG_Pipeline import get_current_qa_chain, get_upload_status
from config import *

#  LLM — used in combine_node to merge both answers
llm = AzureChatOpenAI(
    azure_endpoint    = AZURE_OPENAI_CM_ENDPOINT,
    api_key           = AZURE_OPENAI_CM_API_KEY,
    azure_deployment  = AZURE_OPENAI_CM_DEPLOYMENT_NAME,
    openai_api_version= AZURE_OPENAI_CM_API_VERSION,
    temperature       = TEMPERATURE,
)
#  TAVILY SEARCH TOOL — IPC News + Recent Changes
tavily_tool = TavilySearchResults(
    tavily_api_key      = TAVILY_API_KEY,
    max_results         = 4,
    search_depth        = "advanced",
    include_answer      = True,
    include_raw_content = False,
    name                = "ipc_news_search",
    description         = (
        "Search for recent Indian Penal Code (IPC) news, amendments, "
        "new sections, court judgments, legal updates, and law changes. "
        "Use for current IPC developments and recent legal news."
    ),
)
#  STATE

class IPCState(TypedDict):
    question    : str        # original user question
    rag_answer  : str        # answer from PDF (RAG)
    news_answer : str        # answer from Tavily (recent news)
    final_answer: str        # combined final answer
    messages    : Annotated[list, operator.add]

def rag_node(state: IPCState) -> dict:
    question = state["question"]
    print("\n[IPC Agent] Searching uploaded PDF...")

    qa_chain = get_current_qa_chain()
    if qa_chain is None:
        print("[IPC Agent] No PDF uploaded")
        return {
            "rag_answer": (
                "No IPC document uploaded yet. "
                "Please upload a PDF first."
            )
        }
    result = qa_chain.invoke({"question": question})
    answer = result.get("answer", "No answer found in the document.")

    print("[IPC Agent] Answer retrieved from PDF")

    return {"rag_answer": answer}
# NODE 2 — tavily_node
def tavily_node(state: IPCState) -> dict:

    question = state["question"]
    print("[IPC Agent] Searching recent IPC news...")

    # Simple search query
    search_query = f"Indian Penal Code {question} latest news 2024 2025"

    results = tavily_tool.invoke({"query": search_query})

    # If no results
    if not results:
        return {"news_answer": "No recent IPC news found."}

    # Convert results to simple readable text
    news_text = "Recent IPC News:\n\n"

    for i, r in enumerate(results, 1):
        news_text += f"{i}. {r.get('title')}\n"
        news_text += f"{r.get('content')}\n\n"

    print("[IPC Agent] News retrieved successfully")

    return {"news_answer": news_text}

#  NODE 3 — combine_node
COMBINE_PROMPT = """You are an expert IPC Legal Agent.

You have two sources of information:
  1. Document Answer  → from the uploaded IPC PDF document
  2. Recent News      → from real-time web search (Tavily)

Your job:
  Combine BOTH sources into one clear, complete, well-structured answer.

Structure your answer like this:

FROM THE IPC DOCUMENT:
  [Summarize what the document says — sections, punishments, definitions]

RECENT NEWS & UPDATES:
  [Summarize recent amendments, judgments, news from web search]

COMPLETE ANSWER:
  [Final comprehensive answer combining both sources]

Rules:
- Always cite IPC section numbers from the document
- Mention recent changes or amendments from web search
- If no document is uploaded, say so clearly and answer from news only
- End with: "Note: Please consult a qualified lawyer for legal advice."
"""

def combine_node(state: IPCState) -> dict:
    """
    GPT-4 combines RAG answer + Tavily news into final answer.
    """
    question    = state["question"]
    rag_answer  = state["rag_answer"]
    news_answer = state["news_answer"]

    print(f"[IPC Agent] 💡 Combine Node — GPT-4 merging answers...")

    response = llm.invoke([
        SystemMessage(content=COMBINE_PROMPT),
        HumanMessage(content=(
            f"User Question: {question}\n\n"
            f"--- Source 1: Document Answer ---\n{rag_answer}\n\n"
            f"--- Source 2: Recent News ---\n{news_answer}\n\n"
            f"Please combine both into a final answer."
        )),
    ])

    final_answer = response.content
    print(f"[IPC Agent] ✅ Final answer ready")

    return {
        "final_answer": final_answer,
        "messages"    : [AIMessage(content=final_answer)],
    }


#  BUILD GRAPH

graph_builder = StateGraph(IPCState)

graph_builder.add_node("rag_node",     rag_node)
graph_builder.add_node("tavily_node",  tavily_node)
graph_builder.add_node("combine_node", combine_node)

graph_builder.add_edge(START,          "rag_node")
graph_builder.add_edge("rag_node",     "tavily_node")
graph_builder.add_edge("tavily_node",  "combine_node")
graph_builder.add_edge("combine_node", END)

ipc_graph = graph_builder.compile()

png_data = ipc_graph.get_graph().draw_mermaid_png()
with open('ipc_graph.png', 'wb') as f:
    f.write(png_data)  

#  PUBLIC FUNCTION — called by Orchestrator

def run_ipc_agent(question: str) -> str:

    print(f"\n[IPC Agent] Starting for: '{question}'")

    result = ipc_graph.invoke({
        "question"    : question,
        "rag_answer"  : "",
        "news_answer" : "",
        "final_answer": "",
        "messages"    : [HumanMessage(content=question)],
    })

    return result["final_answer"]
