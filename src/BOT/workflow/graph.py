from langgraph.graph import StateGraph
from src.BOT.entity.state_entity import AgentState
from src.BOT.agents.translation_agent_1 import translation_agent_1
from src.BOT.agents.translation_agent_2 import translation_agent_2
from src.BOT.agents.health_agent import health_agent
from src.BOT.workflow.router import add_graph_edges
from dotenv import load_dotenv
load_dotenv()

# Initialize the graph
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("translation_agent_1", translation_agent_1)
workflow.add_node("health_agent", health_agent)
workflow.add_node("translation_agent_2", translation_agent_2)

# Set the entry point
workflow.set_entry_point("translation_agent_1")

# Add the edges
add_graph_edges(workflow)

# Compile the graph
app = workflow.compile()
