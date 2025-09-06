from src.BOT.entity.state_entity import AgentState
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="gemma2-9b-it", temperature=0)

def translation_agent_2(state: AgentState):


    """Gets the response from the health agent then end the flow for now ,
    further on it will convert the response to the user's language and return it
    users language when identifies , will be stored in state and then passed into this agent"""
    
    print("---TRANSLATION AGENT 2---")
    
    response = state.get('response', '')
    original_language = state.get('original_language', 'English')
    
    print(f"Response to translate: {response[:100]}...")
    print(f"Target language: {original_language}")
    
    prompt = ChatPromptTemplate.from_messages([("system", "Translate the following text to {language}. Just return the translated text and nothing else."), ("human", "{text}")])
    chain = prompt | llm
    result = chain.invoke({"text": response, "language": original_language})
    translated_response = result.content
    
    print(f"Translated response: {translated_response[:100]}...")
    
    return {"response": translated_response}
