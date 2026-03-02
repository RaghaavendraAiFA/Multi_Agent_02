from typing import Annotated
from typing_extensions import TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import operator
import json
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from config import *
from RAG_IPC_agent    import run_ipc_agent
from healthcare_agent import run_healthcare_agent
from software_agent   import run_software_agent
from mongodb_handler  import save_conversation

#  LLM — used by orchestrator + synthesizer nodes
llm = AzureChatOpenAI(
    azure_endpoint    = AZURE_OPENAI_CM_ENDPOINT,
    api_key           = AZURE_OPENAI_CM_API_KEY,
    azure_deployment  = AZURE_OPENAI_CM_DEPLOYMENT_NAME,
    openai_api_version= AZURE_OPENAI_CM_API_VERSION,
    temperature       = TEMPERATURE,   # deterministic routing decisions
)
#  STATE
class OrchestratorState(TypedDict):
    session_id     : str
    question       : str
    selected_agents: list[str]
    agent_answers  : dict
    final_answer   : str
    messages       : Annotated[list, operator.add]

#  AGENT REGISTRY
AGENT_REGISTRY = {
    "ipc"       : run_ipc_agent,
    "healthcare": run_healthcare_agent,
    "software"  : run_software_agent,
}


#  OUT OF SCOPE MESSAGE
OUT_OF_SCOPE_MESSAGE = (
    "I can only answer questions related to:\n"
    "    IPC Legal     — Indian Penal Code, crimes, sections, punishments\n"
    "    Healthcare    — diseases, symptoms, treatments, medications\n"
    "    Software/Tech — programming, frameworks, DevOps, AI/ML\n\n"
    "Please ask a question related to one of these topics. Thank you!"
)

#  ORCHESTRATOR PROMPT
ORCHESTRATOR_PROMPT = """You are an intelligent Orchestrator for a Multi-Agent AI System.
Your only job is to read the user's question and decide which agent(s) should answer it.


AVAILABLE AGENTS:

  "ipc" — Indian Penal Code Legal Agent
    Answers from uploaded IPC PDF document + recent IPC news
    Covers:
    - IPC sections, clauses, amendments (Section 302, 420, 376, etc.)
    - Crimes: murder, theft, assault, fraud, rape, kidnapping, cheating
    - Punishments, fines, imprisonment terms
    - Bailable vs non-bailable offences
    - FIR, chargesheet, bail, cognizable offences
    - Indian criminal law, penal provisions
    - Recent IPC amendments and court judgments
    - Any question with "IPC", "section", "crime", "punishment", "offence", "FIR"

  "healthcare" — Healthcare & Medical Agent
    Covers:
    - Diseases, disorders, conditions (diabetes, cancer, fever, etc.)
    - Symptoms, signs, diagnosis, treatments, procedures
    - Medications, drugs, dosage, side effects
    - Mental health: depression, anxiety, stress, PTSD
    - Nutrition, diet, vitamins, fitness, wellness
    - First aid, emergency care, vaccines
    - Women's health, child health, elderly care
    - Any medical or health-related question

  "software" — Software & Technology Agent
    Covers:
    - All programming languages: Python, JavaScript, Java, C++, Go, Rust, etc.
    - Frontend: React, Next.js, Vue, Angular, HTML, CSS, Tailwind
    - Backend: FastAPI, Django, Flask, Express, Spring Boot
    - Databases: SQL, MongoDB, Redis, PostgreSQL, Vector DBs
    - Cloud & DevOps: AWS, GCP, Azure, Docker, Kubernetes, CI/CD
    - AI / ML / LLM: LangChain, LangGraph, OpenAI, HuggingFace, PyTorch
    - Trending tools, latest releases, new frameworks (any year)
    - Software architecture, design patterns, microservices
    - Cybersecurity, APIs, Git, testing, debugging
    - Any tech question, software question, coding question

ROUTING RULES:

Rule 1 → Single domain   → ONE agent   : ["ipc"] or ["healthcare"] or ["software"]
Rule 2 → Multi domain    → MULTIPLE    : ["ipc", "healthcare"] etc.
Rule 3 → Not in scope    → out_of_scope

OUT OF SCOPE — return "out_of_scope" for:
  Sports, cricket, football | Politics, elections | Weather, travel
  Cooking (unless nutrition) | History | Jokes | Stock market | Movies


EXAMPLES:

"What is IPC Section 302?"                     → ["ipc"]
"Punishment for murder in India?"              → ["ipc"]
"Recent IPC amendments 2024?"                  → ["ipc"]
"What are symptoms of diabetes?"               → ["healthcare"]
"How does async await work in Python?"         → ["software"]
"Trending AI tools in 2026?"                   → ["software"]
"Drug abuse laws and health effects?"          → ["ipc", "healthcare"]
"Cyber crime punishment and how to prevent?"   → ["ipc", "software"]
"Build a healthcare app — legal requirements?" → ["ipc", "healthcare", "software"]
"Who won the cricket match?"                   → out_of_scope
"Tell me a joke"                               → out_of_scope


OUTPUT FORMAT — STRICT:

Return ONLY one of these — no explanation, no extra text:
  ["ipc"]
  ["healthcare"]
  ["software"]
  ["ipc", "healthcare"]
  ["ipc", "software"]
  ["healthcare", "software"]
  ["ipc", "healthcare", "software"]
  out_of_scope
"""
#  SYNTHESIZER PROMPT
SYNTHESIZER_PROMPT = """You are an expert answer synthesizer.
You have received answers from multiple specialized AI agents.
Combine them into ONE clear, well-structured, comprehensive response.

Rules:
1. Merge all answers into a single flowing response
2. Remove any duplication between agents
3. Keep all important facts, section numbers, medical details, tech info
4. Structure clearly with appropriate sections/headers
5. End with: "Note: Please consult a qualified professional for specific advice."
"""
#  NODE 1 — orchestrator_node
def orchestrator_node(state: OrchestratorState) -> dict:
    question = state["question"]
    print(f"\n[Orchestrator] Routing: '{question}'")

    response = llm.invoke([
        SystemMessage(content=ORCHESTRATOR_PROMPT),
        HumanMessage(content=question),
    ])

    raw = response.content.strip()
    print(f"[Orchestrator] Raw response: {raw}")

    if "out_of_scope" in raw.lower():
        return {"selected_agents": ["out_of_scope"]}

    try:
        # Clean and parse JSON list
        clean    = raw.replace("'", '"')
        selected = json.loads(clean)
        selected = [a for a in selected if a in AGENT_REGISTRY]
        if not selected:
            return {"selected_agents": ["out_of_scope"]}
    except Exception:
        return {"selected_agents": ["out_of_scope"]}

    print(f"[Orchestrator] Selected agents: {selected}")
    return {"selected_agents": selected}

#  NODE 2 — run_agents_node
def run_agents_node(state: OrchestratorState) -> dict:
    selected_agents = state["selected_agents"]
    question        = state["question"]
    agent_answers   = {}

    # Out of scope — skip all agents
    if "out_of_scope" in selected_agents:
        return {"agent_answers": {"out_of_scope": OUT_OF_SCOPE_MESSAGE}}

    # Single agent — run directly
    if len(selected_agents) == 1:
        agent_name = selected_agents[0]
        print(f"[Agents] Running single agent: {agent_name}")
        agent_answers[agent_name] = AGENT_REGISTRY[agent_name](question)

    # Multiple agents — run in parallel
    else:
        print(f"[Agents] Running {len(selected_agents)} agents in parallel: {selected_agents}")
        with ThreadPoolExecutor(max_workers=len(selected_agents)) as executor:
            future_to_agent = {
                executor.submit(AGENT_REGISTRY[agent], question): agent
                for agent in selected_agents
            }
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    agent_answers[agent_name] = future.result()
                    print(f"[Agents]  {agent_name} completed")
                except Exception as e:
                    agent_answers[agent_name] = f"Error from {agent_name}: {str(e)}"
                    print(f"[Agents]  {agent_name} failed: {e}")

    return {"agent_answers": agent_answers}

#  NODE 3 — synthesizer_node
def synthesizer_node(state: OrchestratorState) -> dict:
    agent_answers = state["agent_answers"]
    question      = state["question"]
    session_id    = state["session_id"]

    # Out of scope
    if "out_of_scope" in agent_answers:
        final_answer = agent_answers["out_of_scope"]
        print("[Synthesizer] Out-of-scope response")

    # Single agent — use directly
    elif len(agent_answers) == 1:
        final_answer = list(agent_answers.values())[0]
        print("[Synthesizer] Single agent — no synthesis needed")

    # Multiple agents — synthesize
    else:
        print(f"[Synthesizer] Combining {len(agent_answers)} answers...")
        agent_labels = {
            "ipc"       : " IPC Legal Agent",
            "healthcare": " Healthcare Agent",
            "software"  : " Software Agent",
        }
        answers_text = ""
        for name, answer in agent_answers.items():
            label = agent_labels.get(name, name.upper())
            answers_text += f"\n--- {label} ---\n{answer}\n"

        response = llm.invoke([
            SystemMessage(content=SYNTHESIZER_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nAnswers:\n{answers_text}"),
        ])
        final_answer = response.content
        print("[Synthesizer] Synthesis complete!")

    # Save to MongoDB
    try:
        save_conversation(
            session_id  = session_id,
            question    = question,
            answer      = final_answer,
            agents_used = list(agent_answers.keys()),
        )
    except Exception as e:
        print(f"[Synthesizer]  MongoDB save failed: {e}")

    return {
        "final_answer": final_answer,
        "messages"    : [AIMessage(content=final_answer)],
    }

#  BUILD GRAPH
graph_builder = StateGraph(OrchestratorState)

graph_builder.add_node("orchestrator_node", orchestrator_node)
graph_builder.add_node("run_agents_node",   run_agents_node)
graph_builder.add_node("synthesizer_node",  synthesizer_node)

graph_builder.add_edge(START,               "orchestrator_node")
graph_builder.add_edge("orchestrator_node", "run_agents_node")
graph_builder.add_edge("run_agents_node",   "synthesizer_node")
graph_builder.add_edge("synthesizer_node",  END)

orchestrator_graph = graph_builder.compile()

png_data = orchestrator_graph.get_graph().draw_mermaid_png()
with open('orchestrator_graph.png', 'wb') as f:
    f.write(png_data)  

#  PUBLIC FUNCTION — called by Fast_api.py

def run_orchestrator(question: str, session_id: str) -> str:
    result = orchestrator_graph.invoke({
        "session_id"     : session_id,
        "question"       : question,
        "selected_agents": [],
        "agent_answers"  : {},
        "final_answer"   : "",
        "messages"       : [HumanMessage(content=question)],
    })
    return result["final_answer"]