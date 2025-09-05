from langgraph.graph import StateGraph, END
from src.BOT.entity.state_entity import AgentState
from dotenv import load_dotenv
load_dotenv()

def add_graph_edges(graph: StateGraph):
    graph.add_edge("translation_agent_1", "health_agent")
    graph.add_edge("health_agent", "translation_agent_2")
    graph.add_edge("translation_agent_2", END)
