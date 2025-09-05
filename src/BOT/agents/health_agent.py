from src.BOT.entity.state_entity import AgentState
from src.BOT.mcp.api_connector import vector_db_query
from dotenv import load_dotenv
load_dotenv()

def health_agent(state: AgentState):

    """Gets the "translated" query from the translation agent , and passes it to the vector db query tool 
    for now only the vector_db tool later on logic for chaining multiple mcp tools
    will be added on"""

    print("---HEALTH AGENT---")
    query = state['query']
    response = vector_db_query(query)
    return {"response": response}
