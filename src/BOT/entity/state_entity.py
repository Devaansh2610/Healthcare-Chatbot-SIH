from typing_extensions import TypedDict, Annotated, List
from langchain_core.messages import AnyMessage
import operator
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    query: str
    response: str
    original_language: str
