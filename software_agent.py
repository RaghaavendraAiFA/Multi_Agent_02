#  software_agent.py
from typing import Literal
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from config import *

tavily_tool = TavilySearchResults(
    tavily_api_key   = TAVILY_API_KEY,
    max_results      = 3,
    search_depth     = "advanced",
    include_answer   = True,
    include_raw_content = False,
    name             = "tavily_software_search",
    description      = (
        "Search the internet for any software and technology information. "
        "Covers programming languages, frameworks, libraries, databases, "
        "cloud, DevOps, AI/ML, LLMs, trending tools, latest releases, "
        "code examples, bug fixes, best practices, architecture, security."
    ),
)
tools = [tavily_tool]


llm = AzureChatOpenAI(
    azure_endpoint    = AZURE_OPENAI_CM_ENDPOINT,
    api_key           = AZURE_OPENAI_CM_API_KEY,
    azure_deployment  = AZURE_OPENAI_CM_DEPLOYMENT_NAME,
    openai_api_version= AZURE_OPENAI_CM_API_VERSION,
    temperature       = TEMPERATURE,
)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are an elite Software & Technology Expert Agent.

You will receive:
  1. The user's question
  2. Real-time web search results from Tavily (already fetched for you)

Your job:
  Read the search results carefully + use your own expert knowledge
  to give the BEST possible answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR FULL EXPERTISE COVERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROGRAMMING LANGUAGES — every language:
  Python, JavaScript, TypeScript, Java, C, C++, C#, Go, Rust,
  Kotlin, Swift, Dart, PHP, Ruby, Scala, R, MATLAB, Bash, SQL,
  Perl, Haskell, Elixir, Lua, Julia — any language the user asks

FRONTEND DEVELOPMENT:
  React, Next.js, Vue.js, Nuxt, Angular, Svelte, SvelteKit
  HTML5, CSS3, Tailwind CSS, Bootstrap, Material UI, Shadcn/ui
  State management: Redux, Zustand, Pinia, Jotai, Recoil
  Build tools: Vite, Webpack, esbuild, Turbopack, Parcel

BACKEND DEVELOPMENT:
  Python  : FastAPI, Django, Flask, SQLAlchemy, Celery
  Node.js : Express, NestJS, Fastify, Hapi
  Java    : Spring Boot, Micronaut, Quarkus
  Go      : Gin, Fiber, Echo, Chi
  Others  : Rails (Ruby), Laravel (PHP), ASP.NET Core (C#)

DATABASES:
  Relational : PostgreSQL, MySQL, SQLite, SQL Server, Oracle
  NoSQL      : MongoDB, Redis, Cassandra, DynamoDB, Firestore, CouchDB
  Vector DB  : FAISS, Pinecone, Weaviate, Chroma, Qdrant, Milvus
  ORMs       : SQLAlchemy, Prisma, TypeORM, Hibernate, GORM

CLOUD & DEVOPS:
  AWS     : EC2, S3, Lambda, RDS, ECS, EKS, SageMaker, CloudFront
  GCP     : Cloud Run, BigQuery, GKE, Vertex AI, Cloud Functions
  Azure   : App Service, AKS, Cosmos DB, Azure ML, Azure Functions
  Containers: Docker, Kubernetes, Helm, Istio, Podman
  CI/CD   : GitHub Actions, Jenkins, GitLab CI, CircleCI, ArgoCD
  IaC     : Terraform, Ansible, Pulumi, CDK, Crossplane

AI / ML / LLM / AGENTS:
  Frameworks  : LangChain, LangGraph, LlamaIndex, AutoGen, CrewAI, Haystack
  LLM APIs    : OpenAI, Anthropic Claude, Google Gemini, Mistral, Llama, Groq
  ML Libraries: PyTorch, TensorFlow, Keras, JAX, Scikit-learn, XGBoost
  HuggingFace : Transformers, Diffusers, PEFT, TRL, Datasets
  Concepts    : RAG, fine-tuning, embeddings, agents, vector search, prompt engineering
  MLOps       : MLflow, Weights & Biases, DVC, BentoML, Triton, Ray

SOFTWARE ARCHITECTURE:
  Patterns     : Singleton, Factory, Observer, Strategy, Decorator, SOLID
  Architecture : Microservices, Monolith, Serverless, Event-driven, Hexagonal
  APIs         : REST, GraphQL, gRPC, WebSockets, MQTT, tRPC
  Principles   : DDD, CQRS, Event Sourcing, Clean Architecture, Saga pattern

TESTING:
  Unit testing, Integration testing, E2E testing
  pytest, Jest, Vitest, Mocha, JUnit, TestNG
  Playwright, Cypress, Selenium

SECURITY:
  OWASP Top 10, SQL injection, XSS, CSRF, SSRF
  JWT, OAuth2, OpenID Connect, SAML, API keys
  SSL/TLS, mTLS, encryption, hashing, secrets management

TRENDING & LATEST (always current via web search):
  Newest AI models and LLM releases
  Latest framework versions and breaking changes
  Trending tools in 2024, 2025, 2026 and beyond
  Current best practices and industry standards

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO BUILD YOUR ANSWER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Read the Tavily search results provided to you
2. Combine search results with your own expert knowledge
3. Structure your answer clearly:
     - For CONCEPTS     : explain clearly with examples
     - For CODE HELP    : show working code with comments
     - For TRENDING     : list items with descriptions and why they matter
     - For COMPARISONS  : pros/cons table or clear comparison
     - For BUG FIXES    : explain root cause first, then fix it
     - For ARCHITECTURE : explain trade-offs and recommendations
4. Always give practical, production-ready answers
5. Mention if something is from the real-time search results
"""
def software_node(state: MessagesState) -> dict:
    messages = state["messages"]

    # Add system prompt at the top of every call
    response = llm_with_tools.invoke([SYSTEM_PROMPT] + messages)

    # Return as dict — LangGraph appends this to messages list in state
    return {"messages": [response]}


# Built-in LangGraph node — automatically runs whichever tool GPT-4o asked for
tool_node_S = ToolNode(tools)


def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:

    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"   # -> go run Tavily search

    return END 

# BUILD GRAPH

graph_builder = StateGraph(MessagesState)

graph_builder.add_node("software_node", software_node)
graph_builder.add_node("tools", tool_node_S)

graph_builder.add_edge(START,"software_node")
graph_builder.add_conditional_edges(
    "software_node",
    should_continue,
    {
        "tools": "tools",   # if should_continue returns "tools" -> go to tool_node
        END: END            # if should_continue returns END     -> stop
    }

)
graph_builder.add_edge("tools", "software_node")

software_graph = graph_builder.compile()

png_data = software_graph.get_graph().draw_mermaid_png()
with open('software_graph.png', 'wb') as f:
    f.write(png_data)  


def run_software_agent(question: str) -> str:

    print(f"\n[Software Agent] Question: '{question}'")

    result = software_graph.invoke(
        {"messages": [HumanMessage(content=question)]}
    )

    # Last message is GPT-4's final answer
    final_answer = result["messages"][-1].content
    return final_answer

