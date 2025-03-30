import os
import re
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# LangChain imports
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from typing import Optional, List

# Load API keys from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")

if not OPENAI_API_KEY or not FIRECRAWL_API_KEY:
    raise Exception("Missing API keys for OpenAI or Firecrawl!")

app = FastAPI()

# Enable CORS (adjust allow_origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to your specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Firecrawl Integration ---
def firecrawl_scrape(url: str) -> str:
    endpoint = "https://api.firecrawl.dev/scrape"
    headers = {"Authorization": f"Bearer {FIRECRAWL_API_KEY}"}
    payload = {"url": url}
    response = requests.post(endpoint, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data.get("content", "")
    else:
        raise Exception(f"Firecrawl error: {response.text}")

def firecrawl_tool(query: str) -> str:
    try:
        content = firecrawl_scrape(query)
        return f"Scraped content: {content[:500]}..."
    except Exception as e:
        return f"Error scraping URL: {e}"

firecrawl_tool_instance = Tool(
    name="FirecrawlScraper",
    func=firecrawl_tool,
    description="Scrapes a given URL using Firecrawl and returns its content."
)

# --- OpenAI LLM Integration using ChatOpenAI ---
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # or gpt-4 if you have access
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

# --- Build the LangChain Agent ---
# Optionally, you can also include a system prompt in the agent_kwargs if desired.
system_message = (
    "You are a helpful AI assistant. Whenever a query includes a URL, "
    "you should use the FirecrawlScraper tool to fetch the latest content before answering."
)

agent = initialize_agent(
    tools=[firecrawl_tool_instance],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Using the React agent
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={"system_message": system_message}  # Optional: force tool usage when URLs are present
)

# --- FastAPI Endpoint ---
class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_agent(request: QueryRequest):
    try:
        result = agent.run(request.query)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Run the App ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
