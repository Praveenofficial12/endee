"""
main.py — FastAPI Backend Server
=================================
Entry point for the AI Debugging Assistant API.

Endpoints:
  POST /analyze    — Analyse an error/code snippet via the RAG pipeline
  GET  /health     — Health check
  GET  /samples    — Return sample test inputs for the frontend
  GET  /            — Serve the frontend UI
"""

import os
import logging
import sqlite3
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from endee_client import EndeeClient
from rag_pipeline import analyze_error, seed_endee_database

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────
load_dotenv()  # Load .env file if present

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Endee connection settings (from environment or defaults)
ENDEE_URL   = os.getenv("ENDEE_URL", "http://localhost:8080/api/v1")
ENDEE_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")

# Resolve paths
BACKEND_DIR  = Path(__file__).resolve().parent
DATA_PATH    = BACKEND_DIR / "data.json"
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"


# ──────────────────────────────────────────────
#  Lifespan — startup / shutdown
# ──────────────────────────────────────────────
endee_client: EndeeClient = None  # Global reference


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup:
      1. Connect to Endee (or use fallback)
      2. Create index & seed sample data
    On shutdown:
      Cleanup (if needed)
    """
    global endee_client

    logger.info("=" * 60)
    logger.info("  AI Debugging Assistant — Starting Up")
    logger.info("=" * 60)

    # Init SQLite DB for history
    conn = sqlite3.connect("history.db")
    conn.execute('''CREATE TABLE IF NOT EXISTS history 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME, query TEXT, explanation TEXT, fix TEXT)''')
    conn.commit()
    conn.close()

    # Initialise Endee client
    endee_client = EndeeClient(base_url=ENDEE_URL, auth_token=ENDEE_TOKEN)

    # Seed the database with sample error-solution pairs
    try:
        seed_endee_database(endee_client, data_path=str(DATA_PATH))

        if endee_client.is_connected:
            logger.info("✅  Endee database seeded successfully (live connection).")
        else:
            logger.info("✅  Data loaded into fallback in-memory search.")
            logger.info("    ℹ️  Start Endee with Docker for full vector DB features:")
            logger.info("    docker run -p 8080:8080 -v ./endee-data:/data endeeio/endee-server:latest")
    except Exception as e:
        logger.warning("⚠️  Could not seed database: %s", e)
        logger.warning("   The app will still work but results may be limited.")

    # Check OpenAI key
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key and not api_key.startswith("sk-your"):
        logger.info("✅  OpenAI API key configured — LLM analysis enabled.")
    else:
        logger.info("ℹ️  No OpenAI API key — returning best match from knowledge base.")
        logger.info("    Set OPENAI_API_KEY in backend/.env for LLM-powered responses.")

    logger.info("")
    logger.info("🚀  Server is ready!")
    logger.info("   → API:      http://localhost:8000/docs")
    logger.info("   → Frontend: http://localhost:8000")
    logger.info("=" * 60)

    yield  # ← App runs here

    logger.info("Server shutting down.")


# ──────────────────────────────────────────────
#  FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(
    title="AI Debugging App",
    description="RAG-powered code error analyser using Endee Vector Database",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the frontend from any origin (needed for Live Server, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
#  Pydantic Models
# ──────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    """Request body for the /analyze endpoint."""
    input_text: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="The error message, code snippet, or combination to analyse.",
        examples=["TypeError: Cannot read properties of undefined (reading 'map')"],
    )


class AnalyzeResponse(BaseModel):
    """Structured response from the debugging pipeline."""
    explanation: str    = Field(..., description="Root cause explanation")
    fix: str            = Field(..., description="Suggested fix")
    corrected_code: str = Field(..., description="Corrected code snippet")


# ──────────────────────────────────────────────
#  API Endpoints
# ──────────────────────────────────────────────

# In-memory cache for duplicate queries to save LLM/DB latency
QUERY_CACHE: dict = {}

@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_endpoint(request: AnalyzeRequest):
    """
    **Analyse an error or code snippet.**

    Runs the full RAG pipeline:
    1. Generate embedding for the input
    2. Query Endee for similar known errors
    3. Augment the prompt with retrieved context
    4. Call the LLM for a structured response
    """
    input_hash = hash(request.input_text)
    if input_hash in QUERY_CACHE:
        logger.info("Serving response from in-memory cache (0ms latency).")
        return QUERY_CACHE[input_hash]

    try:
        result = analyze_error(
            user_input=request.input_text,
            endee=endee_client,
        )
        response = AnalyzeResponse(**result)

        # Store in SQLite Database history
        try:
            conn = sqlite3.connect("history.db")
            conn.execute("INSERT INTO history (timestamp, query, explanation, fix) VALUES (?, ?, ?, ?)",
                         (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), request.input_text, result["explanation"], result["fix"]))
            conn.commit()
            conn.close()
        except Exception as db_e:
            logger.warning(f"Could not save to history DB: {db_e}")
        
        # Store in cache (limit size to 100 to prevent memory leaks)
        if len(QUERY_CACHE) > 100:
            QUERY_CACHE.clear()
        QUERY_CACHE[input_hash] = response
        
        return response

    except Exception as e:
        logger.exception("Error during analysis")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "AI Debugging Assistant",
        "endee_url": ENDEE_URL,
        "endee_connected": endee_client.is_connected if endee_client else False,
        "llm_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
    }


@app.get("/samples", tags=["Utilities"])
async def get_samples():
    """Return sample error inputs for the frontend's quick-test buttons."""
    return {
        "samples": [
            {
                "label": "Python — TypeError (str + int)",
                "text": "TypeError: can only concatenate str (not \"int\") to str\n\nprint('Score: ' + 100)"
            },
            {
                "label": "JavaScript — undefined .map()",
                "text": "TypeError: Cannot read properties of undefined (reading 'map')\n\nconst data = undefined;\nconst items = data.map(item => item.name);"
            },
            {
                "label": "Java — NullPointerException",
                "text": "NullPointerException\n\nString name = null;\nint length = name.length();"
            },
            {
                "label": "Python — IndexError",
                "text": "IndexError: list index out of range\n\nfruits = ['apple', 'banana', 'cherry']\nprint(fruits[5])"
            },
            {
                "label": "JavaScript — const reassignment",
                "text": "TypeError: Assignment to constant variable\n\nconst PI = 3.14;\nPI = 3.14159;"
            },
            {
                "label": "Python — RecursionError",
                "text": "RecursionError: maximum recursion depth exceeded\n\ndef factorial(n):\n    return n * factorial(n - 1)\n\nprint(factorial(5))"
            },
        ]
    }


@app.get("/database", response_class=HTMLResponse, tags=["Database"])
async def view_database():
    """View the SQLite database history of the app."""
    try:
        conn = sqlite3.connect("history.db")
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, query, explanation, fix FROM history ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()

        html_content = '''
        <html>
            <head>
                <title>Database History</title>
                <style>
                    body { font-family: sans-serif; background: #06080F; color: #e2e8f0; padding: 20px; }
                    h1 { color: #818cf8; }
                    table { width: 100%; border-collapse: collapse; margin-top: 20px; text-align: left; background: #111827; }
                    th, td { border: 1px solid #374151; padding: 12px; }
                    th { background: #1f2937; color: #a78bfa; }
                    td { white-space: pre-wrap; font-size: 0.9em; }
                </style>
            </head>
            <body>
                <h1>AI Debugging App - Database History</h1>
                <table>
                    <tr>
                        <th style="width: 15%">Timestamp</th>
                        <th style="width: 30%">User Query</th>
                        <th style="width: 30%">Root Cause Explanation</th>
                        <th style="width: 25%">Suggested Fix</th>
                    </tr>
        '''
        for r in rows:
            html_content += f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td></tr>"
        
        html_content += '''
                </table>
            </body>
        </html>
        '''
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error reading database</h1><p>{str(e)}</p>")


# ──────────────────────────────────────────────
#  Serve Frontend (static files + SPA fallback)
# ──────────────────────────────────────────────

# Serve individual frontend files with correct content types
@app.get("/style.css", include_in_schema=False)
async def serve_css():
    """Serve the CSS file."""
    css_path = FRONTEND_DIR / "style.css"
    if css_path.exists():
        return FileResponse(str(css_path), media_type="text/css")
    raise HTTPException(404, "CSS file not found")


@app.get("/script.js", include_in_schema=False)
async def serve_js():
    """Serve the JavaScript file."""
    js_path = FRONTEND_DIR / "script.js"
    if js_path.exists():
        return FileResponse(str(js_path), media_type="application/javascript")
    raise HTTPException(404, "JS file not found")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the main frontend page."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    # If no frontend directory, return a simple message
    return HTMLResponse(
        "<h1>AI Debugging Assistant</h1>"
        "<p>Frontend not found. API is running at <a href='/docs'>/docs</a></p>",
        status_code=200,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
