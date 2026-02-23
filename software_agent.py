#  SOFTWARE AGENT — Built with LangGraph + OpenAI GPT-4o
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from config import *

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_CM_ENDPOINT,
    api_key=AZURE_OPENAI_CM_API_KEY, 
    azure_deployment= AZURE_OPENAI_CM_DEPLOYMENT_NAME,  # or your deployed chat model
    openai_api_version=AZURE_OPENAI_CM_API_VERSION ,
    temperature=TEMPERATURE 
)

# SYSTEM PROMPT
# Defines the full scope of the Software Agent.

SYSTEM_PROMPT = SystemMessage(content="""You are an expert Software Agent — a senior-level software engineer and architect 
with deep, up-to-date knowledge across the entire software development landscape.

YOUR KNOWLEDGE COVERS:

PROGRAMMING LANGUAGES:
  - Python, JavaScript, TypeScript, Java, C, C++, C#
  - Go, Rust, Kotlin, Swift, Dart, Ruby, PHP, Scala
  - R, MATLAB, SQL, Bash/Shell scripting

FRONTEND:
  - React, Next.js, Vue.js, Nuxt, Angular, Svelte
  - HTML5, CSS3, Tailwind CSS, Bootstrap
  - State management: Redux, Zustand, Pinia
  - Build tools: Vite, Webpack, esbuild

BACKEND:
  - Python: FastAPI, Django, Flask, SQLAlchemy
  - Node.js: Express, NestJS, Fastify
  - Java: Spring Boot, Micronaut
  - Go: Gin, Fiber
  - Ruby on Rails, Laravel (PHP), ASP.NET Core

DATABASES:
  - Relational: PostgreSQL, MySQL, SQLite, SQL Server
  - NoSQL: MongoDB, Redis, Cassandra, DynamoDB
  - Vector DBs: Pinecone, Weaviate, Chroma, FAISS
  - ORMs: SQLAlchemy, Prisma, TypeORM, Hibernate

CLOUD & DEVOPS:
  - AWS (EC2, S3, Lambda, RDS, ECS, EKS)
  - GCP (Cloud Run, BigQuery, GKE)
  - Azure (App Service, AKS, Cosmos DB)
  - Docker, Kubernetes, Helm
  - CI/CD: GitHub Actions, Jenkins, GitLab CI
  - Terraform, Ansible, Pulumi

AI / ML / LLM:
  - LangChain, LangGraph, LlamaIndex
  - OpenAI API, Anthropic Claude, Google Gemini
  - HuggingFace, Transformers, Diffusers
  - PyTorch, TensorFlow, Keras, Scikit-learn
  - RAG, fine-tuning, embeddings, vector search
  - MLflow, Weights & Biases, Vertex AI

ARCHITECTURE & DESIGN:
  - Design patterns: Singleton, Factory, Observer, Strategy, etc.
  - Microservices, Monolith, Serverless, Event-driven
  - REST, GraphQL, gRPC, WebSockets
  - Domain-Driven Design (DDD), CQRS, Event Sourcing
  - Clean Architecture, Hexagonal Architecture

TESTING:
  - Unit, Integration, E2E testing
  - pytest, Jest, Vitest, Mocha, JUnit
  - Playwright, Cypress, Selenium

SECURITY:
  - OWASP Top 10, SQL injection, XSS, CSRF
  - JWT, OAuth2, OpenID Connect
  - SSL/TLS, encryption, hashing

LATEST TRENDS (2024-2025):
  - AI-assisted coding (GitHub Copilot, Cursor, Devin)
  - LLM Agents and multi-agent systems
  - Edge computing and serverless evolution
  - WebAssembly (WASM) adoption
  - Bun and Deno as Node.js alternatives
  - Rust growing in systems and web
  - Platform engineering and internal developer platforms
  - OpenTelemetry for observability


HOW YOU RESPOND:

1. Give accurate, practical, production-level answers
2. Always include working code examples with comments
3. Mention multiple approaches when relevant + recommend the best one
4. Explain trade-offs (pros/cons) for architecture decisions
5. For code bugs/errors: explain the root cause, then fix it
6. Keep code clean, readable, and following best practices
7. Mention version-specific differences if they matter
8. For trending topics: give honest, balanced perspective
""")
#   NODE
#  Flow: START -> software_node -> END

def software_node(state: MessagesState) -> dict:
    messages = state["messages"]

    # Prepend system prompt so GPT-4o knows its role on every call
    response = llm.invoke([SYSTEM_PROMPT] + messages)

    return {"messages": [response]}

#  BUILD THE GRAPH

graph_builder = StateGraph(MessagesState)

# Add the single node
graph_builder.add_node("software_node", software_node)

# Add edges: START -> software_node -> END
graph_builder.add_edge(START, "software_node")
graph_builder.add_edge("software_node", END)

# Compile into a runnable graph
software_graph = graph_builder.compile()

png_data = software_graph.get_graph().draw_mermaid_png()
with open('software_graph.png', 'wb') as f:
    f.write(png_data)

#  STEP 6 — PUBLIC FUNCTION

def run_software_agent(question: str) -> str:
    print("\n  Software Agent activated...")

    result = software_graph.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    # Last message in state = GPT-4o's final answer
    final_answer = result["messages"][-1].content
    return final_answer


# to Test  
if __name__ == "__main__":
    print("=" * 60)
    print("       Software Agent  LangGraph Test")
    print("=" * 60)
    print("Ask about: languages, frameworks, architecture,")
    print("DevOps, AI/ML, trends, code help, debugging...")
    print("=" * 60)

    while True:
        question = input("\nAsk a software question (or 'quit'): ").strip()

        if question.lower() == "quit":
            print("Exiting Software Agent.")
            break

        if not question:
            continue

        answer = run_software_agent(question)
        print("\nAnswer:")
        print("-" * 60)
        print(answer)
        print("=" * 60)
