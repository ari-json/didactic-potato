import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# LangChain parts
from langchain.agents import initialize_agent, Tool
from langchain.llms.base import LLM
from typing import Optional, List

# Load API keys from environment variables
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")

if not OPENROUTER_API_KEY or not FIRECRAWL_API_KEY:
    raise Exception("Missing API keys for OpenRouter or Firecrawl!")

app = FastAPI()

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

# --- OpenRouter LLM Integration ---
def openrouter_call(prompt: str) -> str:
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek/deepseek-r1-zero:free",  # Using DeepSeek R1 Zero (free)
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    response = requests.post(endpoint, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        raise Exception(f"OpenRouter error: {response.text}")

class OpenRouterLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return openrouter_call(prompt)

llm = OpenRouterLLM()

# --- Build the LangChain Agent ---
agent = initialize_agent(
    tools=[firecrawl_tool_instance],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
