# software_agent.py
from typing import Literal
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from config import *
#  TAVILY SEARCH TOOL
tavily_tool = TavilySearchResults(
    tavily_api_key=TAVILY_API_KEY,
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    name="tavily_software_search",
    description=(
        "Search the internet for real-time software and technology information. "
        "Use this for questions about latest framework versions, recent library updates, "
        "new language features, trending tools, current best practices, "
        "recent tech news, newest AI models, and any software topic that may have "
        "changed or been released recently."
    ),
)

tools = [tavily_tool]


#  LLM
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_CM_ENDPOINT,
    api_key= AZURE_OPENAI_CM_API_KEY,
    azure_deployment=AZURE_OPENAI_CM_DEPLOYMENT_NAME,
    openai_api_version=AZURE_OPENAI_CM_API_VERSION,
    temperature=TEMPERATURE,
)

# Binding tools tells GPT-4: "you CAN call these tools if needed"
llm_with_tools = llm.bind_tools(tools)


# SYSTEM PROMPT

SYSTEM_PROMPT = SystemMessage(content="""You are an expert Software & Technology Agent — a senior software engineer 
with deep knowledge across the entire software development landscape.


WHEN TO USE TAVILY SEARCH TOOL:

ALWAYS call tavily_software_search when the question involves ANY of these:
  - Words: "trending", "latest", "newest", "recent", "current", "top", "best in"
  - Words: "just released", "new version", "updated", "what's new"
  - Any year mentioned: 2024, 2025, 2026 or beyond
  - AI/ML trends, LLM updates, new model releases
  - New frameworks, libraries, tools released recently
  - Current best practices that change frequently
  - "what are the trends", "what is popular now", "what tools are used"
  - Technology news or recent developments

YOU MUST SEARCH — do NOT answer from memory for these — your training data is outdated.
Always prefer fresh Tavily results for anything trend-related.

ANSWER DIRECTLY from memory (no search) ONLY for:
  - Core programming concepts: loops, functions, OOP, recursion
  - Established patterns: REST basics, SQL basics, design patterns
  - How well-known frameworks fundamentally work (not version-specific)
  - Debugging help or code review of user-provided code


YOUR KNOWLEDGE COVERS:

Languages     : Python, JavaScript, TypeScript, Java, C/C++, Go, Rust, Kotlin, Swift
Frontend      : React, Next.js, Vue, Angular, Svelte, Tailwind CSS
Backend       : FastAPI, Django, Flask, Express, NestJS, Spring Boot
Databases     : PostgreSQL, MongoDB, Redis, FAISS, Pinecone, DynamoDB
Cloud & DevOps: AWS, GCP, Azure, Docker, Kubernetes, GitHub Actions, Terraform
AI / ML       : LangChain, LangGraph, OpenAI API, HuggingFace, PyTorch, TensorFlow
Architecture  : Microservices, Serverless, Event-driven, DDD, CQRS, Clean Architecture
Security      : JWT, OAuth2, OWASP, SSL/TLS


HOW YOU RESPOND:

1. For ALL trending/year-based questions → ALWAYS call Tavily first, then combine with your knowledge
2. For general concepts → answer directly with clear explanation
3. Always include working code examples with comments when relevant
4. Mention pros/cons for architecture decisions
5. When using Tavily results → clearly present the real-time findings
""")


#TOOL NODE

tool_node = ToolNode(tools)


#  SHOULD CONTINUE

def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"    # GPT-4 wants to search Tavily
    return END            # GPT-4 gave a final answer


# SOFTWARE NODE

def software_node(state: MessagesState) -> dict:
    messages  = state["messages"]
    response  = llm_with_tools.invoke([SYSTEM_PROMPT] + messages)

    # ── ADD THIS to see if tool was called ──
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"[Software Agent] ✅ Tavily SEARCH triggered!")
        for tc in response.tool_calls:
            print(f"[Software Agent] 🔍 Search query: {tc['args'].get('query', '')}")
    else:
        print(f"[Software Agent] ✅ Answering from LLM knowledge (no search)")

    return {"messages": [response]}

#  BUILD GRAPH

graph_builder = StateGraph(MessagesState)

graph_builder.add_node("software_node", software_node)
graph_builder.add_node("tools",         tool_node)

graph_builder.add_edge(START, "software_node")

graph_builder.add_conditional_edges(
    "software_node",
    should_continue,
    {
        "tools": "tools",   # search needed → go to tool_node
        END:     END,       # final answer  → stop
    }
)

# After tool_node runs → loop back to software_node
# GPT-4 reads search results and writes final answer
graph_builder.add_edge("tools", "software_node")

software_graph = graph_builder.compile()


# PUBLIC FUNCTION

def run_software_agent(question: str) -> str:

    print(f"\n[Software Agent] Processing: {question}")

    result = software_graph.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    final_answer = result["messages"][-1].content
    return final_answer


#  TEST — python software_agent.py

if __name__ == "__main__":
    print("=" * 55)
    print("  Software Agent — LLM + Tavily Search")
    print("=" * 55)
    print("Try trending: 'latest Python version features'")
    print("Try general : 'how does async await work'")
    print("=" * 55)

    while True:
        question = input("\nAsk a software question (or 'quit'): ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue
        answer = run_software_agent(question)
        print(f"\nAnswer:\n{'-' * 55}\n{answer}\n{'=' * 55}")