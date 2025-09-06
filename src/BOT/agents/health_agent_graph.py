from typing import List
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AnyMessage, ToolMessage, SystemMessage
import operator
import json
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

class MultiToolAgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

llm = ChatGroq(model="gemma2-9b-it", temperature=0)

async def tool_calling_llm(state: MultiToolAgentState):
    with open('src/BOT/mcp/mcp_config.json', 'r') as f:
        mcp_config = json.load(f)

    client = MultiServerMCPClient(connections=mcp_config["mcpServers"])
    tools = await client.get_tools()
    
    tool_descriptions = "\n".join([f'{tool.name}: {tool.description}' for tool in tools])
    system_message_content = f"""You are a helpful AI assistant with access to these tools:
{tool_descriptions}

IMPORTANT: Route queries to the appropriate tool based on the query type:

1. HEALTH QUERIES (symptoms, medical conditions, health concerns):
   - Use: symptom_checker_tool
   - Examples: "I have a sore throat", "What are symptoms of flu", "I feel dizzy"

2. WORK/INTERNSHIP/EXPERIENCE QUERIES:
   - Use: vector_db_query  
   - Examples: "Tell me about the health guidlines or anything related to health guidlines", "According to the health guidlines what should I do in this situation", "give a sumary of the health guidlines "

3. OTHER QUERIES:
   - If the query doesn't fit the above categories, respond directly without using tools.

Given a user query, determine the appropriate tool to use and call it with the relevant parameters."""
    system_message = SystemMessage(content=system_message_content)
    
    messages = [system_message] + state["messages"]
    
    llm_with_tools = llm.bind_tools(tools)
    
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

async def dynamic_tool_node(state: MultiToolAgentState):
    tool_calls = state["messages"][-1].tool_calls
    
    with open('src/BOT/mcp/mcp_config.json', 'r') as f:
        mcp_config = json.load(f)

    client = MultiServerMCPClient(connections=mcp_config["mcpServers"])
    all_tools = await client.get_tools()
    
    tool_messages = []
    for tool_call in tool_calls:
        selected_tool = next((tool for tool in all_tools if tool.name == tool_call["name"]), None)
        if selected_tool:
            output = await selected_tool.ainvoke(tool_call["args"])
            tool_messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
    
    return {"messages": tool_messages}

def tools_condition(state: MultiToolAgentState) -> str:
    last_message = state["messages"][-1]
    
    # If we have a tool message, we're done
    if isinstance(last_message, ToolMessage):
        return END
    
    # If the LLM made tool calls, execute them
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # If no tool calls were made, we're done (LLM provided direct response)
    return END

# Create the graph
multi_tool_graph_builder = StateGraph(MultiToolAgentState)

# Add the nodes
multi_tool_graph_builder.add_node("tool_calling_llm", tool_calling_llm)
multi_tool_graph_builder.add_node("tools", dynamic_tool_node)

# Set the entry point
multi_tool_graph_builder.set_entry_point("tool_calling_llm")

# Add the conditional edges
multi_tool_graph_builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
)

multi_tool_graph_builder.add_edge("tools", "tool_calling_llm")

# Compile the graph
multi_tool_app = multi_tool_graph_builder.compile()
