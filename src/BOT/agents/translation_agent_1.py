from src.BOT.entity.state_entity import AgentState
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="gemma2-9b-it", temperature=0)

def translation_agent_1(state: AgentState):

    """Gets the query from the user , and passes it to the health agent 
    for now it will retur nothing , just put back the user query in state , later on we will use the commented out
    google translation api mcp tool in api_connector to translate user query to english , and then pass it to health_agent """

    print("---TRANSLATION AGENT 1---")
    
    # Get the last message content properly
    last_message = state['messages'][-1]
    if hasattr(last_message, 'content'):
        query = last_message.content
    elif isinstance(last_message, dict):
        # Handle dictionary format
        query = last_message.get('content', str(last_message))
    elif isinstance(last_message, tuple):
        # Handle tuple format (role, content)
        query = last_message[1]
    else:
        query = str(last_message)
    
    print(f"Original query: {query}")
    
    # Detect language
    lang_prompt = ChatPromptTemplate.from_messages([("system", "Detect the language of the following text. Just return the language name and nothing else."), ("human", "{query}")])
    lang_chain = lang_prompt | llm
    lang_result = lang_chain.invoke({"query": query})
    original_language = lang_result.content
    
    # Translate query to English
    trans_prompt = ChatPromptTemplate.from_messages([("system", "Translate the following user query to English. Just return the translated query and nothing else."), ("human", "{query}")])
    trans_chain = trans_prompt | llm
    trans_result = trans_chain.invoke({"query": query})
    translated_query = trans_result.content
    
    print(f"Translated query: {translated_query}")
    print(f"Original language: {original_language}")
    
    return {"query": translated_query, "original_language": original_language}
