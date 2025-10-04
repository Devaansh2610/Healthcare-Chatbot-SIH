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

IMPORTANT: Route queries to the appropriate tool based on the query type.
This docstring provides detailed instructions to ensure the correct tool
is used for each type of query and to prevent misrouting.

1. HEALTH QUERIES (physical symptoms, illnesses, personal health concerns):
   - Tool: symptom_checker_tool
   - When to use: User describes physical symptoms, asks about possible medical conditions,
     or wants advice about what might be causing their health issues.
     Includes symptoms like fever, cough, headache, nausea, dizziness, rash, fatigue, etc.
   - Examples:
       "I have a sore throat and fever"
       "What are the symptoms of flu?"
       "I feel dizzy and nauseous"
   - Notes: Only use for personal symptom-related queries. Do not use for mental health,
     diet, exercise, or general guidelines queries.

2. MENTAL HEALTH QUERIES (stress, anxiety, sadness, burnout, emotional concerns):
   - Tool: mental_health_management_tool
   - When to use: User expresses emotional or mental struggles, stress management,
     or general wellbeing guidance. Includes anxiety, depression, burnout, low motivation, etc.
   - Examples:
       "I am feeling anxious and overwhelmed"
       "I can't sleep because of stress from work"
       "I feel depressed and lonely"
   - Notes: Only use for emotional or mental wellbeing concerns. Include disclaimers about
     seeking professional help if needed. Do not route physical symptom queries here.

3. NUTRITION AND DIET QUERIES:
   - Tool: nutrition_advice_tool
   - When to use: User asks about diet plans, meal suggestions, nutritional advice,
     or foods to include/avoid. Can include preferences (vegetarian, likes Indian food)
     and dietary restrictions (gluten-free, lactose-intolerant).
   - Examples:
       "I want to lose weight, what should I eat?"
       "Give me a vegetarian meal plan for increasing protein intake"
       "What should I eat to increase iron while being gluten-free?"
   - Notes: Only use for diet/nutrition-related goals. Do not use for physical symptoms,
     mental health, or exercise guidance.

4. EXERCISE AND FITNESS QUERIES:
   - Tool: exercise_suggestions_tool
   - When to use: User asks about workouts, fitness routines, improving stamina, strength,
     or general exercise guidance. Can include fitness level and physical restrictions.
   - Examples:
       "I want to build strength at home"
       "Suggest exercises for someone with back pain"
       "I am a beginner and want to improve stamina"
   - Notes: Only use for physical activity or fitness goals. Do not use for diet,
     mental health, or symptom queries.

5. GENERAL HEALTH GUIDELINES QUERIES:
   - Tool: vector_db_query
   - When to use: User asks about official health guidelines, workplace policies,
     documented recommendations, or general health protocol summaries.
   - Examples:
       "Tell me about the health guidelines in our region"
       "According to the health guidelines, what should I do?"
       "Give me a summary of the health guidelines"
   - Notes: Do not route personal symptom, diet, exercise, or mental health queries here.

6. OTHER QUERIES (unrelated to health, diet, fitness, or mental wellbeing):
   - Tool: None (respond directly without using any MCP tool)
   - When to use: Query does not fit into any of the above categories (general knowledge,
     casual conversation, or unrelated questions).
   - Examples:
       "What is the weather today?"
       "Tell me a joke"
       "Who won the football match yesterday?"
   - Notes: Avoid forcing a tool.
   - Response: As a health chatbot, politely inform the user that this query is not health-related.
   - Example response: "I am a health-focused assistant, so I can only help with health, diet, fitness, or mental wellbeing queries."

Key Rules to Avoid Wrong Tool Selection:
----------------------------------------
- Physical symptoms → symptom_checker_tool
- Mental/emotional concerns → mental_health_management_tool
- Diet/nutrition → nutrition_advice_tool
- Fitness/exercise → exercise_suggestions_tool
- Guidelines/policy → vector_db_query
- Everything else → respond directly

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
