from src.BOT.entity.state_entity import AgentState
from src.BOT.agents.health_agent_graph import multi_tool_app
from dotenv import load_dotenv
load_dotenv()

async def health_agent(state: AgentState):

    """Gets the "translated" query from the translation agent, and invokes the multi-tool agent subgraph that can handle both health and work/internship queries."""

    print("---MULTI-TOOL AGENT---")
    query = state.get('query', '')
    
    if not query:
        print("No query found in state, using fallback")
        # Fallback to last message if query is not in state
        last_message = state['messages'][-1]
        if hasattr(last_message, 'content'):
            query = last_message.content
        elif isinstance(last_message, dict):
            query = last_message.get('content', str(last_message))
        elif isinstance(last_message, tuple):
            query = last_message[1]
        else:
            query = str(last_message)
    
    print(f"Processing query: {query}")
    
    # Create the initial state for the health agent subgraph
    initial_state = {"messages": [("human", query)]}
    
    # Invoke the multi-tool agent subgraph
    final_state = await multi_tool_app.ainvoke(initial_state)
    
    # Get the final response from the subgraph
    response = final_state["messages"][-1].content
    
    print(f"Multi-tool agent response: {response[:100]}...")
    
    return {"response": response}
