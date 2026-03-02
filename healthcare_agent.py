#  HEALTHCARE AGENT
from typing import Literal
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from config import * 

# Setup Tavily Search Tool

tavily_tool = TavilySearch(
    tavily_api_key=TAVILY_API_KEY,
    max_results=3,
    search_depth="advanced",
    include_answer=True,           # Tavily returns a direct answer too
    include_raw_content=False,
    name="tavily_health_search",
    description=(
        "Search the internet for real-time health and medical information. "
        "Use this for any question about diseases, symptoms, treatments, "
        "medications, wellness tips, or medical procedures."
    ),
)

tools = [tavily_tool]

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_CM_ENDPOINT,
    api_key=AZURE_OPENAI_CM_API_KEY,
    azure_deployment= AZURE_OPENAI_CM_DEPLOYMENT_NAME,  # or your deployed chat model
    openai_api_version=AZURE_OPENAI_CM_API_VERSION ,
    temperature=TEMPERATURE 
)

llm_with_tools = llm.bind_tools(tools)
# STEP 4: System Prompt for the Healthcare Agent
SYSTEM_PROMPT = SystemMessage(content="""You are a professional Healthcare Agent with access to real-time medical search.

Your responsibilities:
- Answer questions about diseases, symptoms, causes, and treatments
- Provide information on medications, dosages, and side effects
- Give preventive healthcare and wellness advice
- Explain medical procedures and tests
- Cover mental health topics and nutrition advice

How to respond:
1. ALWAYS use the search tool to get up-to-date medical information
2. Summarize search results clearly and in simple language
3. Use bullet points to list symptoms, treatments, prevention tips
4. For emergencies, immediately tell user to call emergency services
5. End every response with: "Please consult a qualified doctor for personal medical advice."
""")


def healthcare_node(state: MessagesState) -> dict:
    messages = state["messages"]

    # Add system prompt at the top of every call
    response = llm_with_tools.invoke([SYSTEM_PROMPT] + messages)

    # Return as dict — LangGraph appends this to messages list in state
    return {"messages": [response]}


# Built-in LangGraph node — automatically runs whichever tool GPT-4o asked for
tool_node = ToolNode(tools)


def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:

    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"   # -> go run Tavily search

    return END 

graph_builder = StateGraph(MessagesState)

# Add nodes to the graph
graph_builder.add_node("healthcare_node", healthcare_node)
graph_builder.add_node("tools", tool_node)

# Edge: START -> healthcare_node
graph_builder.add_edge(START, "healthcare_node")

# Conditional edge: healthcare_node -> tools OR END
graph_builder.add_conditional_edges(
    "healthcare_node",
    should_continue,
    {
        "tools": "tools",   # if should_continue returns "tools" -> go to tool_node
        END: END            # if should_continue returns END     -> stop
    }
)

# Edge: tool_node -> back to healthcare_node (loop)
graph_builder.add_edge("tools", "healthcare_node")

# Compile the graph into a runnable agent
healthcare_graph = graph_builder.compile()

png_data = healthcare_graph.get_graph().draw_mermaid_png()
with open('healthcare_graph.png', 'wb') as f:
    f.write(png_data)  

#Public fun  

def run_healthcare_agent(question: str) -> str:
    print("\n  Healthcare Agent activated...")
    print(f"  Searching Tavily for: {question}\n")

    result = healthcare_graph.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    # Last message in state = final AI answer
    final_answer = result["messages"][-1].content
    return final_answer
