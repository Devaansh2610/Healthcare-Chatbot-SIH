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
    Query the local FastAPI vector DB endpoint.
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
    Symptom Checker MCP Tool (LLM wrapper).
    Dynamically analyzes symptoms using the LLM.

    Args:
        symptoms: Free-text symptoms description, e.g. "I have sore throat and fever".
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