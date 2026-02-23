# # ============================================================
# #  orchestrator.py
# # ============================================================

# from typing import Annotated
# from typing_extensions import TypedDict
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import operator
# import json

# from langchain_openai import AzureChatOpenAI
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langgraph.graph import StateGraph, START, END

# from config import *

# from RAG_IPC_agent    import run_ipc_agent
# from healthcare_agent import run_healthcare_agent
# from software_agent   import run_software_agent
# from mongodb_handler  import save_conversation


# # ─────────────────────────────────────────────────────
# #  LLM
# # ─────────────────────────────────────────────────────
# llm = AzureChatOpenAI(
#     azure_endpoint=AZURE_CHAT_ENDPOINT,
#     api_key=AZURE_CHAT_API_KEY,
#     azure_deployment=AZURE_CHAT_DEPLOYMENT,
#     openai_api_version=AZURE_CHAT_API_VERSION,
#     temperature=TEMPERATURE,
# )


# # ─────────────────────────────────────────────────────
# #  STATE
# # ─────────────────────────────────────────────────────
# class OrchestratorState(TypedDict):
#     session_id:      str
#     question:        str
#     selected_agents: list[str]
#     agent_answers:   dict
#     final_answer:    str
#     messages:        Annotated[list, operator.add]


# # ─────────────────────────────────────────────────────
# #  AGENT REGISTRY
# # ─────────────────────────────────────────────────────
# AGENT_REGISTRY = {
#     "ipc":        run_ipc_agent,
#     "healthcare": run_healthcare_agent,
#     "software":   run_software_agent,
# }


# # ─────────────────────────────────────────────────────
# #  NODE 1 — ORCHESTRATOR
# # ─────────────────────────────────────────────────────
# ORCHESTRATOR_PROMPT = """You are an orchestrator that decides which AI agents should answer a user question.

# Available agents:
# - "ipc"        : Indian Penal Code, legal sections, crimes, punishments, FIR, bail
# - "healthcare" : diseases, symptoms, treatments, medications, health, wellness
# - "software"   : programming, code, frameworks, DevOps, AI/ML, software architecture

# Rules:
# 1. Pick ONE agent if the question clearly belongs to one domain
# 2. Pick MULTIPLE agents if the question spans more than one domain
# 3. Always return a valid JSON list — nothing else

# Examples:
#   "What is IPC Section 302?"                           → ["ipc"]
#   "What are symptoms of diabetes?"                     → ["healthcare"]
#   "How do I use FastAPI?"                              → ["software"]
#   "Drug abuse laws and its health effects?"            → ["ipc", "healthcare"]

# Return ONLY the JSON list. No explanation. No extra text.
# """

# def orchestrator_node(state: OrchestratorState) -> dict:
#     question = state["question"]
#     print(f"\n[Orchestrator] Routing: '{question}'")

#     response = llm.invoke([
#         SystemMessage(content=ORCHESTRATOR_PROMPT),
#         HumanMessage(content=question),
#     ])

#     raw = response.content.strip()

#     try:
#         clean    = raw.replace("```json", "").replace("```", "").strip()
#         selected = json.loads(clean)
#         selected = [a for a in selected if a in AGENT_REGISTRY]
#         if not selected:
#             selected = ["software"]
#     except Exception:
#         print(f"[Orchestrator] Parse error — defaulting to software. Raw: {raw}")
#         selected = ["software"]

#     print(f"[Orchestrator] Selected agents: {selected}")
#     return {"selected_agents": selected}


# # ─────────────────────────────────────────────────────
# #  NODE 2 — RUN AGENTS (Parallelization)
# # ─────────────────────────────────────────────────────
# def run_agents_node(state: OrchestratorState) -> dict:
#     question        = state["question"]
#     selected_agents = state["selected_agents"]
#     agent_answers   = {}

#     if len(selected_agents) == 1:
#         agent_name = selected_agents[0]
#         print(f"[Agents] Running single agent: {agent_name}")
#         agent_answers[agent_name] = AGENT_REGISTRY[agent_name](question)

#     else:
#         print(f"[Agents] Running in PARALLEL: {selected_agents}")
#         with ThreadPoolExecutor(max_workers=len(selected_agents)) as executor:
#             future_to_agent = {
#                 executor.submit(AGENT_REGISTRY[agent], question): agent
#                 for agent in selected_agents
#             }
#             for future in as_completed(future_to_agent):
#                 agent_name = future_to_agent[future]
#                 try:
#                     agent_answers[agent_name] = future.result()
#                     print(f"[Agents] ✓ {agent_name} completed")
#                 except Exception as e:
#                     agent_answers[agent_name] = f"Error: {str(e)}"
#                     print(f"[Agents] ✗ {agent_name} failed: {e}")

#     return {"agent_answers": agent_answers}


# # ─────────────────────────────────────────────────────
# #  NODE 3 — SYNTHESIZER + MONGODB SAVE
# # ─────────────────────────────────────────────────────
# SYNTHESIZER_PROMPT = """You are a synthesizer. Multiple AI agents have answered the same question.

# Your job:
# 1. Combine all answers into ONE clear, unified response
# 2. Do not repeat information — merge overlapping points
# 3. Separate sections clearly if topics are different
# 4. Keep all important details from each agent
# 5. Make it easy to read and understand
# """

# def synthesizer_node(state: OrchestratorState) -> dict:
#     agent_answers = state["agent_answers"]
#     question      = state["question"]
#     session_id    = state["session_id"]

#     if len(agent_answers) == 1:
#         final_answer = list(agent_answers.values())[0]
#         print("[Synthesizer] Single answer — no synthesis needed")

#     else:
#         print(f"[Synthesizer] Combining {len(agent_answers)} answers...")

#         agent_labels = {
#             "ipc":        "IPC Legal Agent",
#             "healthcare": "Healthcare Agent",
#             "software":   "Software Agent",
#         }

#         answers_text = ""
#         for agent_name, answer in agent_answers.items():
#             label = agent_labels.get(agent_name, agent_name.upper())
#             answers_text += f"\n--- {label} ---\n{answer}\n"

#         response = llm.invoke([
#             SystemMessage(content=SYNTHESIZER_PROMPT),
#             HumanMessage(content=f"Question: {question}\n\nAnswers:\n{answers_text}\n\nSynthesize:"),
#         ])

#         final_answer = response.content
#         print("[Synthesizer] Done!")

#     # Save to MongoDB
#     save_conversation(
#         session_id  = session_id,
#         question    = question,
#         answer      = final_answer,
#         agents_used = list(agent_answers.keys()),
#     )

#     return {
#         "final_answer": final_answer,
#         "messages":     [AIMessage(content=final_answer)],
#     }


# # ─────────────────────────────────────────────────────
# #  BUILD GRAPH
# # ─────────────────────────────────────────────────────
# graph_builder = StateGraph(OrchestratorState)

# graph_builder.add_node("orchestrator_node", orchestrator_node)
# graph_builder.add_node("run_agents_node",   run_agents_node)
# graph_builder.add_node("synthesizer_node",  synthesizer_node)

# graph_builder.add_edge(START,               "orchestrator_node")
# graph_builder.add_edge("orchestrator_node", "run_agents_node")
# graph_builder.add_edge("run_agents_node",   "synthesizer_node")
# graph_builder.add_edge("synthesizer_node",  END)

# orchestrator_graph = graph_builder.compile()

# png_data = orchestrator_graph.get_graph().draw_mermaid_png()
# with open('orchestrator_graph.png', 'wb') as f:
#     f.write(png_data)
# # ─────────────────────────────────────────────────────
# #  EXPOSED FUNCTION — called by Fast_api.py
# #
# #  Takes both question AND session_id
# # ─────────────────────────────────────────────────────
# def run_orchestrator(question: str, session_id: str) -> str:
#     """
#     Args:
#         question   : user's question
#         session_id : unique session ID from Fast_api.py

#     Returns:
#         str: final answer
#     """
#     initial_state: OrchestratorState = {
#         "session_id":      session_id,
#         "question":        question,
#         "selected_agents": [],
#         "agent_answers":   {},
#         "final_answer":    "",
#         "messages":        [HumanMessage(content=question)],
#     }

#     result = orchestrator_graph.invoke(initial_state)
#     return result["final_answer"]

# ============================================================
#  orchestrator.py
# ============================================================

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


# ─────────────────────────────────────────────────────
#  LLM
# ─────────────────────────────────────────────────────
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_CM_ENDPOINT,
    api_key=AZURE_OPENAI_CM_API_KEY,
    azure_deployment=AZURE_OPENAI_CM_DEPLOYMENT_NAME,
    openai_api_version=AZURE_OPENAI_CM_API_VERSION,
    temperature=TEMPERATURE,

)


# ─────────────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────────────
class OrchestratorState(TypedDict):
    session_id:      str
    question:        str
    selected_agents: list[str]
    agent_answers:   dict
    final_answer:    str
    messages:        Annotated[list, operator.add]


# ─────────────────────────────────────────────────────
#  AGENT REGISTRY
# ─────────────────────────────────────────────────────
AGENT_REGISTRY = {
    "ipc":        run_ipc_agent,
    "healthcare": run_healthcare_agent,
    "software":   run_software_agent,
}

# Message shown when question is out of scope
OUT_OF_SCOPE_MESSAGE = (
    "I can only answer questions related to:\n"
    "   IPC Legal        — Indian Penal Code, crimes, sections, punishments\n"
    "   Healthcare       — diseases, symptoms, treatments, medications\n"
    "   Software & Tech  — programming, frameworks, AI & ML \n\n"
    "Please ask a question related to one of these topics. Thank you!"
)


#  NODE 1 — ORCHESTRATOR
ORCHESTRATOR_PROMPT = """You are an orchestrator that decides which AI agents should handle a user question.

Available agents:
- "ipc"        : Indian Penal Code, legal sections, crimes, punishments, FIR, bail, Indian criminal law
- "healthcare" : diseases, symptoms, treatments, medications, health, wellness, medical procedures
- "software"   : programming, code, frameworks, DevOps, cloud, AI/ML, software architecture, tech

Rules:
1. Pick ONE agent if the question clearly belongs to one domain
2. Pick MULTIPLE agents if the question spans more than one domain
3. If the question is NOT related to any of the above domains, return exactly: "out_of_scope"
4. Return ONLY a valid JSON list OR the string "out_of_scope" — nothing else

Examples:
  "What is IPC Section 302?"                           → ["ipc"]
  "What are symptoms of diabetes?"                     → ["healthcare"]
  "How do I use FastAPI?"                              → ["software"]
  "Drug abuse laws and its health effects?"            → ["ipc", "healthcare"]
  "What is the weather today?"                         → out_of_scope
  "Who won the cricket match?"                         → out_of_scope
  "Tell me a joke"                                     → out_of_scope
  "What is the capital of France?"                     → out_of_scope
  "Legal way to sell medicines and build the website?" → ["ipc", "healthcare", "software"]

Return ONLY the JSON list or out_of_scope. No explanation. No extra text.
"""

def orchestrator_node(state: OrchestratorState) -> dict:

    question = state["question"]
    print(f"\n[Orchestrator] Routing: '{question}'")

    response = llm.invoke([
        SystemMessage(content=ORCHESTRATOR_PROMPT),
        HumanMessage(content=question),
    ])

    raw = response.content.strip()
    print(f"[Orchestrator] Raw response: {raw}")

    # Check if out of scope
    if "out_of_scope" in raw.lower():
        print("[Orchestrator] Question is out of scope")
        return {"selected_agents": ["out_of_scope"]}

    # Parse JSON list of agents
    try:
        clean    = raw.replace("```json", "").replace("```", "").strip()
        selected = json.loads(clean)

        # Keep only valid known agents
        selected = [a for a in selected if a in AGENT_REGISTRY]

        # If nothing valid found after filtering → out of scope
        if not selected:
            print("[Orchestrator] No valid agents found — marking out of scope")
            return {"selected_agents": ["out_of_scope"]}

    except Exception:
        print(f"[Orchestrator] Parse error — marking out of scope. Raw: {raw}")
        return {"selected_agents": ["out_of_scope"]}

    print(f"[Orchestrator] Selected agents: {selected}")
    return {"selected_agents": selected}


# ─────────────────────────────────────────────────────
#  NODE 2 — RUN AGENTS (Parallelization)
#
#  If "out_of_scope" in selected_agents → skip running any agent
#  Single agent  → run directly
#  Multiple agents → run ALL in parallel via ThreadPoolExecutor
# ─────────────────────────────────────────────────────
def run_agents_node(state: OrchestratorState) -> dict:

    question        = state["question"]
    selected_agents = state["selected_agents"]
    agent_answers   = {}

    # Skip if out of scope — no agent needs to run
    if "out_of_scope" in selected_agents:
        print("[Agents] Out of scope — skipping agent execution")
        return {"agent_answers": {"out_of_scope": OUT_OF_SCOPE_MESSAGE}}

    if len(selected_agents) == 1:
        # Single agent — run directly
        agent_name = selected_agents[0]
        print(f"[Agents] Running single agent: {agent_name}")
        agent_answers[agent_name] = AGENT_REGISTRY[agent_name](question)

    else:
        # Multiple agents — run ALL in parallel
        print(f"[Agents] Running in PARALLEL: {selected_agents}")
        with ThreadPoolExecutor(max_workers=len(selected_agents)) as executor:
            future_to_agent = {
                executor.submit(AGENT_REGISTRY[agent], question): agent
                for agent in selected_agents
            }
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    agent_answers[agent_name] = future.result()
                    print(f"[Agents] ✓ {agent_name} completed")
                except Exception as e:
                    agent_answers[agent_name] = f"Error: {str(e)}"
                    print(f"[Agents] ✗ {agent_name} failed: {e}")

    return {"agent_answers": agent_answers}


# ─────────────────────────────────────────────────────
#  NODE 3 — SYNTHESIZER + MONGODB SAVE
#
#  Out of scope  → return polite message directly, save to MongoDB
#  Single agent  → return answer directly, no synthesis needed
#  Multi agents  → GPT-4 combines all answers into one response
# ─────────────────────────────────────────────────────
SYNTHESIZER_PROMPT = """You are a synthesizer. Multiple AI agents have answered the same question.

Your job:
1. Combine all answers into ONE clear, unified response
2. Do not repeat information — merge overlapping points
3. Separate sections clearly if topics are different
4. Keep all important details from each agent
5. Make it easy to read and understand
"""

def synthesizer_node(state: OrchestratorState) -> dict:
    """
    Combines answers or returns out-of-scope message.
    Saves everything to MongoDB.
    """
    agent_answers = state["agent_answers"]
    question      = state["question"]
    session_id    = state["session_id"]

    # ── Out of scope ──
    if "out_of_scope" in agent_answers:
        final_answer = agent_answers["out_of_scope"]
        print("[Synthesizer] Returning out-of-scope message")

    # ── Single agent ──
    elif len(agent_answers) == 1:
        final_answer = list(agent_answers.values())[0]
        print("[Synthesizer] Single answer — no synthesis needed")

    # ── Multiple agents — synthesize ──
    else:
        print(f"[Synthesizer] Combining {len(agent_answers)} answers...")

        agent_labels = {
            "ipc":        "IPC Legal Agent",
            "healthcare": "Healthcare Agent",
            "software":   "Software Agent",
        }

        answers_text = ""
        for agent_name, answer in agent_answers.items():
            label = agent_labels.get(agent_name, agent_name.upper())
            answers_text += f"\n--- {label} ---\n{answer}\n"

        response = llm.invoke([
            SystemMessage(content=SYNTHESIZER_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nAnswers:\n{answers_text}\n\nSynthesize:"),
        ])

        final_answer = response.content
        print("[Synthesizer] Synthesis complete!")

    # ── Save to MongoDB ──
    save_conversation(
        session_id  = session_id,
        question    = question,
        answer      = final_answer,
        agents_used = list(agent_answers.keys()),
    )

    return {
        "final_answer": final_answer,
        "messages":     [AIMessage(content=final_answer)],
    }


# ─────────────────────────────────────────────────────
#  BUILD GRAPH
#
#  START → orchestrator_node → run_agents_node → synthesizer_node → END
# ─────────────────────────────────────────────────────
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
# ─────────────────────────────────────────────────────
#  EXPOSED FUNCTION — called by Fast_api.py
# ─────────────────────────────────────────────────────
def run_orchestrator(question: str, session_id: str) -> str:
    """
    Main entry point called by Fast_api.py.

    Args:
        question   : user's question
        session_id : unique session ID

    Returns:
        str: final answer or out-of-scope message
    """
    initial_state: OrchestratorState = {
        "session_id":      session_id,
        "question":        question,
        "selected_agents": [],
        "agent_answers":   {},
        "final_answer":    "",
        "messages":        [HumanMessage(content=question)],
    }

    result = orchestrator_graph.invoke(initial_state)
    return result["final_answer"]