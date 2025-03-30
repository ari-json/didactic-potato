import os
import re
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Load API keys from environment variables
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
if not OPENROUTER_API_KEY or not FIRECRAWL_API_KEY:
    raise Exception("Missing API keys for OpenRouter or Firecrawl!")

# --- Firecrawl Scraping Function ---
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

# --- OpenRouter Call Function ---
def openrouter_call(prompt: str) -> str:
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek/deepseek-r1-zero:free",  # Use the free DeepSeek model
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    response = requests.post(endpoint, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        raise Exception(f"OpenRouter error: {response.text}")

# --- Helper to Extract URL from Query ---
def extract_url(query: str) -> str:
    # Use a simple regex to extract the first URL found in the query
    match = re.search(r'https?://\S+', query)
    return match.group(0) if match else None

# --- Build a Prompt for the LLM ---
def build_prompt(query: str, scraped_content: str) -> str:
    # Instruct the LLM to extract all pricing details from the scraped content
    prompt = f"""User Query: {query}

The following text is the content scraped from the URL:
{scraped_content}

Please extract and list all pricing information from the content in a clear and concise manner.
"""
    return prompt

# --- API Request Model ---
class QueryRequest(BaseModel):
    query: str

# --- API Endpoint ---
@app.post("/ask")
async def ask_scraper(request: QueryRequest):
    query = request.query
    url = extract_url(query)
    if not url:
        raise HTTPException(status_code=400, detail="No URL found in the query")
    try:
        # Scrape the webpage using Firecrawl
        scraped_content = firecrawl_scrape(url)
        # Build a prompt combining the query and scraped content
        prompt = build_prompt(query, scraped_content)
        # Call the LLM via OpenRouter to extract pricing info
        answer = openrouter_call(prompt)
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
