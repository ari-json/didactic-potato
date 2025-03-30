import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# LangChain imports
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI  # Use the ChatOpenAI model from LangChain

# Optional: Enable CORS if you need cross-origin requests (e.g., from a frontend)
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API keys from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")

if not OPENAI_API_KEY or not FIRECRAWL_API_KEY:
    raise Exception("Missing API keys for OpenAI or Firecrawl!")

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

# --- OpenAI LLM Integration ---
# Use your regular OpenAI API key with the ChatOpenAI model
llm = ChatOpenAI(
    temperature=0, 
    model_name="gpt-3.5-turbo",  # Change to another model if desired
    openai_api_key=OPENAI_API_KEY
)

# --- Build the LangChain Agent ---
agent = initialize_agent(
    tools=[firecrawl_tool_instance],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True  # Handles any output parsing errors gracefully
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
