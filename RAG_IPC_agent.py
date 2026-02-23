#  ipc_agent.py

from typing import Annotated
from IPython.display import display
from IPython import display
from openai.types import Image
from typing_extensions import TypedDict
import operator
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from IPC_RAG_Pipeline import get_ipc_qa_chain

qa_chain = get_ipc_qa_chain()

class IPCState(TypedDict):
    messages: Annotated[list, operator.add]

def ipc_node(state: IPCState) -> dict:
    # Get the latest user question from state
    last_message = state["messages"][-1]
    question = last_message.content

    print(f"IPC Agent processing: {question}")

    # Call the QA chain — it handles FAISS retrieval + memory + LLM internally
    result = qa_chain.invoke({"question": question})
    answer = result["answer"]

    # Return answer as AIMessage to be added to state
    return {"messages": [AIMessage(content=answer)]}

#  BUILD GRAPH

graph_builder = StateGraph(IPCState)

graph_builder.add_node("ipc_node", ipc_node)
graph_builder.add_edge(START, "ipc_node")
graph_builder.add_edge("ipc_node", END)

ipc_graph = graph_builder.compile()


png_data = ipc_graph.get_graph().draw_mermaid_png()
with open('ipc_graph.png', 'wb') as f:
    f.write(png_data)  


#Agent 

 
def run_ipc_agent(question: str) -> str:

    result = ipc_graph.invoke(
        {"messages": [HumanMessage(content=question)]}
    )
    return result["messages"][-1].content

#  RUN DIRECTLY TO TEST

if __name__ == "__main__":
    print("=" * 50)
    print("  IPC Agent — Test Mode")
    print("=" * 50)

    while True:
        question = input("\nAsk an IPC question (or 'quit'): ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue
        answer = run_ipc_agent(question)
        print(f"\nAnswer:\n{answer}")