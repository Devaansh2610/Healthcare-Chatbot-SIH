import httpx
import os
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any
from typing_extensions import TypedDict, Annotated, List
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()



llm = ChatGroq(model="gemma2-9b-it", temperature=0)

BASE_URL_1 = "http://127.0.0.1:8000"

API_HOST = f"{BASE_URL_1}/query"

# Create MCP server
mcp = FastMCP(name="api_connector", host="0.0.0.0", port=8010)


@mcp.tool()
def vector_db_query(query: str) -> dict:
    """
    Query the local FastAPI server , for health guidlines related queries , anything related to health guidelines 
    """
    response = httpx.get(API_HOST, params={"q": query})
    return response.json()


# @mcp.tool()
# def google_translate_text(text: str, target_language: str, source_language: str = None) -> dict:
#     """
#     Translates text using the Google Translate API.
#     Requires GOOGLE_PROJECT_ID and GOOGLE_API_KEY environment variables.
#     """
#     project_id = os.environ.get("GOOGLE_PROJECT_ID")
#     api_key = os.environ.get("GOOGLE_API_KEY")

#     if not project_id or not api_key:
#         return {"error": "GOOGLE_PROJECT_ID and GOOGLE_API_KEY environment variables must be set."}

#     url = f"https://translate.googleapis.com/v3beta1/projects/{project_id}:translateText?key={api_key}"

#     request_body = {
#         "contents": [text],
#         "targetLanguageCode": target_language,
#     }

#     if source_language:
#         request_body["sourceLanguageCode"] = source_language

#     try:
#         with httpx.Client() as client:
#             response = client.post(url, json=request_body, timeout=10.0)
#             response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
#             return response.json()
#     except httpx.RequestError as exc:
#         return {"error": f"An error occurred while requesting from Google Translate API: {exc}"}
#     except Exception as exc:
#         return {"error": f"An unexpected error occurred: {exc}"}




@mcp.tool()
def symptom_checker_tool(symptoms: str) -> Dict[str, Any]:
    """
    When 
    Dynamically analyzes symptoms using the LLM.
    Use tool when user is talking anything about if they have any symotoms.
    Args:
        symptoms: Free-text symptoms description.
    """

    if not symptoms or not symptoms.strip():
        return {"error": "No symptoms provided."}

    # Build system + user prompt inline
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful healthcare assistant. "
                "When given symptoms, your task is to:\n"
                "1. List possible common conditions that might explain them.\n"
                "2. Provide short, patient-friendly explanations for each.\n"
                "3. End with general safe advice on when to consult a doctor.\n\n"
                "⚠️ Important: This is general information only, NOT medical advice."
            )
        },
        {
            "role": "user",
            "content": f"My symptoms are: {symptoms}"
        }
    ]

    llm_response = llm.invoke(messages)

    return {
        "analysis": llm_response.content,
        "disclaimer": "This is general information only, not a medical diagnosis."
    }





@mcp.tool()
def mental_health_management_tool(concern: str) -> Dict[str, Any]:
    """
    Supports users with mental health concerns by providing coping strategies, 
    self-care routines, and guidance on when to seek professional help.
    Use this tool when the user expresses stress, anxiety, sadness, burnout, 
    or other mental/emotional struggles.

    Args:
        concern: Free-text description of the user's mental/emotional concern.
    """

    if not concern or not concern.strip():
        return {"error": "No concern provided."}

    # Build system + user prompt inline
    messages = [
        {
            "role": "system",
            "content": (
                "You are a compassionate and supportive mental health assistant. "
                "When given a concern, your job is to:\n"
                "1. Acknowledge and validate the user’s feelings.\n"
                "2. Suggest healthy coping strategies (breathing, journaling, routines, etc.).\n"
                "3. Provide safe, general self-care recommendations (sleep, exercise, social support).\n"
                "4. End with clear guidance on when to seek professional help or crisis support.\n\n"
                "⚠️ Important: You are NOT a therapist. Provide general support only. "
                "If the user mentions self-harm or severe distress, advise them to seek immediate professional help."
            )
        },
        {
            "role": "user",
            "content": f"My concern is: {concern}"
        }
    ]

    llm_response = llm.invoke(messages)

    return {
        "support_plan": llm_response.content,
        "disclaimer": (
            "This is general wellbeing support and not a substitute for professional "
            "mental health care. If you are in crisis or considering self-harm, please seek "
            "help immediately from a qualified professional or call your local emergency number."
        )
    }


@mcp.tool()
def nutrition_advice_tool(goal: str, preferences: str = "", restrictions: str = "") -> Dict[str, Any]:
    """
    Provides personalized nutrition and diet guidance.
    Use this tool when the user asks about diet, food choices, or nutrition.
    
    Args:
        goal: The user’s health or diet goal (e.g., "weight loss", "build muscle", "increase iron intake").
        preferences: (Optional) Food preferences (e.g., "vegetarian", "likes Indian food").
        restrictions: (Optional) Dietary restrictions (e.g., "lactose intolerant", "gluten-free").
    """

    if not goal or not goal.strip():
        return {"error": "No goal provided."}

    # Build system + user prompt inline
    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly and knowledgeable nutrition assistant. "
                "When given a user’s health goal, preferences, and restrictions, you should:\n"
                "1. Suggest a few meal or snack ideas tailored to the goal.\n"
                "2. Recommend foods to include and foods to avoid.\n"
                "3. Give simple, practical tips for daily nutrition.\n"
                "4. Always respect dietary restrictions.\n\n"
                "⚠️ Important: This is general information only, not professional medical or dietary advice."
            )
        },
        {
            "role": "user",
            "content": (
                f"My goal is: {goal}\n"
                f"My preferences: {preferences}\n"
                f"My restrictions: {restrictions}"
            )
        }
    ]

    llm_response = llm.invoke(messages)

    return {
        "nutrition_plan": llm_response.content,
        "disclaimer": "This is general nutrition information only, not a substitute for professional dietary advice."
    }



@mcp.tool()
def exercise_suggestions_tool(goal: str, fitness_level: str = "beginner", restrictions: str = "") -> Dict[str, Any]:
    """
    Provides personalized exercise suggestions.
    Use this tool when the user asks about workouts, fitness, or staying active.
    
    Args:
        goal: The user’s fitness/health goal (e.g., "weight loss", "strength", "better stamina").
        fitness_level: User’s current level (e.g., "beginner", "intermediate", "advanced").
        restrictions: (Optional) Any physical restrictions or injuries (e.g., "knee pain", "back issue").
    """

    if not goal or not goal.strip():
        return {"error": "No fitness goal provided."}

    messages = [
        {
            "role": "system",
            "content": (
                "You are a supportive fitness assistant. "
                "When given a fitness goal, level, and restrictions, you should:\n"
                "1. Suggest safe, suitable exercises (home or gym options).\n"
                "2. Provide a short routine or sample workout.\n"
                "3. Suggest frequency and intensity in simple terms.\n"
                "4. Adapt recommendations if the user mentions injuries or restrictions.\n\n"
                "⚠️ Important: This is general fitness guidance only, not a replacement for professional medical or fitness advice."
            )
        },
        {
            "role": "user",
            "content": (
                f"My fitness goal is: {goal}\n"
                f"My fitness level: {fitness_level}\n"
                f"My restrictions: {restrictions}"
            )
        }
    ]

    llm_response = llm.invoke(messages)

    return {
        "exercise_plan": llm_response.content,
        "disclaimer": "This is general fitness guidance only, not a substitute for professional advice."
    }


# @mcp.tool()
# def lingvanex_translate_text(text: str, to_lang: str, from_lang: str = 'en') -> dict:
#     """
#     Translates text using the Lingvanex Translate API.
#     Requires LINGVANEX_API_KEY environment variables.
#     """
#     api_key = os.environ.get("LINGVANEX_API_KEY")

#     if not api_key:
#         return {"error": "LINGVANEX_API_KEY environment variable must be set."}

#     url = "https://api-b2b.backenster.com/b1/api/v3/translate"
    
#     headers = {
#         'Authorization': f'Bearer {api_key}',
#         'Content-Type': 'application/json'
#     }

#     data = {
#         'from': from_lang,
#         'to': to_lang,
#         'text': text,
#         'platform': 'api'
#     }

#     try:
#         with httpx.Client() as client:
#             response = client.post(url, json=data, headers=headers, timeout=10.0)
#             response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
#             return response.json()
#     except httpx.RequestError as exc:
#         return {"error": f"An error occurred while requesting from Lingvanex API: {exc}"}
#     except Exception as exc:
#         return {"error": f"An unexpected error occurred: {exc}"}







if __name__ == "__main__":
    mcp.run(transport="streamable-http")