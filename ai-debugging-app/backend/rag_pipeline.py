"""
rag_pipeline.py — RAG (Retrieval-Augmented Generation) Pipeline
================================================================
Orchestrates the full debugging workflow:
  1. Generate an embedding for the user's error/code input
  2. Query Endee for semantically similar error-solution pairs
  3. Build a context-augmented prompt from retrieved documents
  4. Send the prompt to the LLM (OpenAI-compatible)
  5. Return structured { explanation, fix, corrected_code }

Works in three modes:
  • Full mode:     Endee + OpenAI  (best results)
  • Partial mode:  Endee only      (returns best match from knowledge base)
  • Fallback mode: Local search    (in-memory cosine similarity + local dataset)
"""

import os
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

from endee_client import EndeeClient

# ──────────────────────────────────────────────
#  Logger
# ──────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Embedding Model (local, no API key needed)
# ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"     # 384-dim, fast, great quality
embedding_model = None                         # Lazy-loaded


def _get_embedding_model():
    """Lazy-load the SentenceTransformer model (downloads once, ~80 MB)."""
    global embedding_model
    if embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model '%s' ...", EMBEDDING_MODEL_NAME)
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded successfully.")
        except ImportError:
            logger.error("sentence-transformers not installed! Install with: pip install sentence-transformers")
            raise RuntimeError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
    return embedding_model


def generate_embedding(text: str) -> List[float]:
    """
    Convert a text string into a 384-dimensional embedding vector.

    Args:
        text: The error message, code snippet, or combined input.

    Returns:
        A list of 384 floats representing the semantic embedding.
    """
    model = _get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


# ──────────────────────────────────────────────
#  Dataset Seeding
# ──────────────────────────────────────────────
_dataset_cache: List[Dict] = []   # Cached for fallback use


def seed_endee_database(endee: EndeeClient, data_path: str = "data.json") -> None:
    """
    Read the sample error-solution dataset, generate embeddings,
    and upsert them into the Endee vector database.

    Args:
        endee:     An initialised EndeeClient instance.
        data_path: Path to the JSON dataset file.
    """
    global _dataset_cache

    # Ensure the index exists
    endee.create_index_if_not_exists()

    # Load dataset
    data_file = Path(data_path)
    if not data_file.exists():
        logger.error("Dataset file not found: %s", data_path)
        return

    with open(data_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    _dataset_cache = dataset   # Cache for fallback
    logger.info("Seeding %d error-solution records ...", len(dataset))

    records = []
    for entry in dataset:
        # Combine error + code for a richer embedding
        combined_text = f"{entry['error']} {entry.get('error_code', '')}"
        vector = generate_embedding(combined_text)

        records.append({
            "id": entry["id"],
            "vector": vector,
            "meta": {
                "language":       entry["language"],
                "error":          entry["error"],
                "error_code":     entry.get("error_code", ""),
                "explanation":    entry["explanation"],
                "fix":            entry["fix"],
                "corrected_code": entry["corrected_code"],
            },
            "filter": {
                "language": entry["language"]
            }
        })

    endee.upsert_vectors(records)
    logger.info("Database seeding complete — %d vectors stored.", len(records))


# ──────────────────────────────────────────────
#  Context Builder
# ──────────────────────────────────────────────
def _build_context(results: List[Dict[str, Any]]) -> str:
    """
    Format the retrieved Endee results into a readable context block
    that will be injected into the LLM prompt.
    """
    if not results:
        return "No similar errors found in the knowledge base."

    context_parts = []
    for i, r in enumerate(results, 1):
        meta = r.get("meta", {})
        similarity = r.get("similarity", 0)
        context_parts.append(
            f"--- Similar Error #{i} (similarity: {similarity:.3f}) ---\n"
            f"Language:  {meta.get('language', 'N/A')}\n"
            f"Error:     {meta.get('error', 'N/A')}\n"
            f"Code:      {meta.get('error_code', 'N/A')}\n"
            f"Root Cause: {meta.get('explanation', 'N/A')}\n"
            f"Fix:       {meta.get('fix', 'N/A')}\n"
            f"Corrected: {meta.get('corrected_code', 'N/A')}\n"
        )
    return "\n".join(context_parts)


def _detect_language(text: str) -> str:
    """Heuristic logic to guess the programming language of the input."""
    text_lower = text.lower()
    if 'def ' in text or 'print(' in text or 'import ' in text or 'elif ' in text:
        return "Python"
    if 'console.log' in text or 'const ' in text or 'let ' in text or '=>' in text or 'function(' in text:
        return "JavaScript"
    if 'public class' in text or 'system.out.println' in text or 'string args' in text:
        return "Java"
    if '#include' in text or 'std::cout' in text or 'int main()' in text or 'namespace ' in text:
        return "C++"
    if 'fmt.print' in text or 'func main()' in text or ':=' in text:
        return "Go"
    if 'fn main()' in text or 'println!' in text or 'let mut ' in text:
        return "Rust"
    return "Unknown"


def _extract_best_match(results: List[Dict[str, Any]], user_input: str = "") -> Dict[str, str]:
    """
    Extract the single best matching result and return it
    as a structured response. Used when no LLM is available.
    Also applies heuristics if the similarity is very low.
    """
    if not results:
        return {
            "explanation": "No matching errors found in the knowledge base.",
            "fix": "Try describing the error differently or include the code snippet.",
            "corrected_code": "# No matching solution found"
        }

    best = results[0]
    meta = best.get("meta", {})
    similarity = best.get("similarity", 0)
    match_lang = meta.get("language", "Unknown")
    user_lang = _detect_language(user_input)

    # Heuristic fallback for common syntax issues
    if similarity < 0.85 and user_input:
        # Python: Missing colon after if/def/for/while
        if user_lang == "Python":
            lines = user_input.split('\n')
            for i, line in enumerate(lines):
                stripped = line.strip()
                if (stripped.startswith(('if ', 'def ', 'for ', 'while ', 'elif ')) or stripped == 'else') and not stripped.endswith(':'):
                    return {
                        "explanation": f"[Heuristic Analysis — Python]\n\nA control flow statement (if, def, for, etc.) is missing a colon ':' at the end of the line.",
                        "fix": "Append a colon ':' to the end of the statement header.",
                        "corrected_code": user_input.replace(line, line + ':')
                    }
        
        # Missing quotes in print/log
        if ('print(' in user_input or 'console.log(' in user_input) and '")' not in user_input and "')" not in user_input and '`)' not in user_input:
             return {
                 "explanation": f"[Heuristic Analysis — {user_lang}]\n\nThe code looks like a print/log statement that is missing quotes around the string.",
                 "fix": "Add quotes around the text inside the print/log function.",
                 "corrected_code": (
                     user_input.replace('print(', 'print("').replace(')', '")') if user_lang == "Python"
                     else user_input.replace('console.log(', 'console.log("').replace(')', '")')
                 )
             }
        
        # Unclosed braces
        if '{' in user_input and '}' not in user_input:
             return {
                 "explanation": f"[Heuristic Analysis — {user_lang}]\n\nThe code contains an opening curly brace '{{' but is missing the corresponding closing brace '}}'.",
                 "fix": "Add a closing curly brace '}' to properly close the code block.",
                 "corrected_code": f"{user_input}\n}}"
             }
        
        # Missing semicolons in C-style languages
        if user_lang in ["C++", "Java", "JavaScript"] and '\n' in user_input:
             lines = [line.strip() for line in user_input.split('\n') if line.strip()]
             for i, line in enumerate(lines):
                 if line and not line.endswith(';') and not line.endswith('{') and not line.endswith('}') and not line.startswith('#') and not line.startswith('/'):
                     if any(word in line for word in ["cout", "println", "return", "=", "fetch", "await"]):
                         return {
                             "explanation": f"[Heuristic Analysis — {user_lang}]\n\nThe code appears to be missing a semicolon ';' at the end of a statement.",
                             "fix": f"Add a semicolon ';' to the end of the line.",
                             "corrected_code": user_input.replace(line, line + ';')
                         }

    # If the similarity is high and the language matches, return the actual corrected code from the database
    if similarity >= 0.80 and (user_lang == match_lang or user_lang == "Unknown"):
        return {
            "explanation": (
                f"[Matched from Knowledge Base — similarity: {similarity:.1%}]\n\n"
                f"{meta.get('explanation', 'No explanation available.')}"
            ),
            "fix": meta.get("fix", "No fix available."),
            "corrected_code": meta.get('corrected_code', user_input)
        }

    # If the user inputted code in a different language than the match, DO NOT show mismatched generic code.
    # Instead, we will annotate their own input code.
    if user_lang != "Unknown" and user_lang != match_lang:
        return {
            "explanation": (
                f"[Concept Match — {match_lang} logic applied to {user_lang}]\n\n"
                f"{meta.get('explanation', 'No explanation available.')}"
            ),
            "fix": meta.get("fix", "Review the required syntax for this implementation."),
            "corrected_code": f"// The database match was in {match_lang}.\n// Proposed fix for your {user_lang} code:\n\n{user_input}\n\n// AI Suggestion: {meta.get('fix', '')}"
        }

    return {
        "explanation": (
            f"[Related Insight — similarity: {similarity:.1%}]\n\n"
            f"{meta.get('explanation', 'No explanation available.')}"
        ),
        "fix": meta.get("fix", "No fix available."),
        "corrected_code": f"/* AI Suggestion based on {match_lang} match:\n * {meta.get('fix', 'None')}\n */\n\n{user_input}",
    }


# ──────────────────────────────────────────────
#  LLM Call
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert programming debugger and educator.
You will receive a user's error message or code snippet, along with
similar known error-solution pairs retrieved from a knowledge base.

Your job is to:
1. **Identify the language**: Automatically determine which programming language the user's code is written in (Python, JS, C++, Java, Rust, Go, etc.).
2. **Explain**: State the root cause of the error clearly and concisely.
3. **Suggest a fix**: Provide a short, actionable description of what to change.
4. **Provide corrected code**: Output the full corrected version of the code IN THE EXACT SAME LANGUAGE AS THE INPUT.

IMPORTANT: You MUST respond with ONLY a valid JSON object in this exact format:
{
  "explanation": "<root cause explanation>",
  "fix": "<actionable fix description>",
  "corrected_code": "<the corrected code snippet IN THE SAME LANGUAGE>"
}

Do NOT include markdown fences, extra text, or anything outside the JSON object."""


def _call_llm(user_input: str, context: str) -> Dict[str, str]:
    """
    Send the context-augmented prompt to the LLM and parse the response.
    Returns fallback if no API key is set.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

    if not api_key or api_key.startswith("sk-your"):
        # No valid API key — skip LLM call
        logger.info("No valid OPENAI_API_KEY — skipping LLM call.")
        return None   # Signal to use RAG-only response

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        user_prompt = (
            f"## User's Error / Code:\n{user_input}\n\n"
            f"## Similar Known Errors from Knowledge Base:\n{context}\n\n"
            "Analyse the error and respond with the JSON object."
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0]

        return json.loads(raw)

    except ImportError:
        logger.warning("openai package not installed — skipping LLM call.")
        return None
    except json.JSONDecodeError:
        logger.error("LLM returned invalid JSON: %s", raw[:200])
        return {
            "explanation": raw,
            "fix": "See explanation above.",
            "corrected_code": ""
        }
    except Exception as e:
        logger.warning("LLM call failed: %s — using RAG-only response.", e)
        return None


# ──────────────────────────────────────────────
#  Public API — Main Pipeline Entry Point
# ──────────────────────────────────────────────
def analyze_error(user_input: str, endee: EndeeClient) -> Dict[str, str]:
    """
    Full RAG pipeline:
      1. Embed the user input
      2. Search Endee (or fallback) for similar errors
      3. Build context from results
      4. Call LLM with augmented prompt (if API key available)
      5. Return structured response

    Args:
        user_input: The error message or code snippet.
        endee:      An initialised EndeeClient.

    Returns:
        { "explanation": "...", "fix": "...", "corrected_code": "..." }
    """
    logger.info("=" * 50)
    logger.info("Analyzing input (%d chars) ...", len(user_input))

    # Step 1 — Generate embedding for the query
    logger.info("Step 1: Generating embedding ...")
    query_vector = generate_embedding(user_input)

    # Step 2 — Search Endee for top-3 similar errors
    logger.info("Step 2: Searching Endee for similar errors ...")
    results = endee.search(query_vector=query_vector, top_k=3)

    # Step 3 — Build context string from results
    context = _build_context(results)
    logger.info("Step 3: Context built from %d retrieved results.", len(results))

    # Step 4 — Call the LLM with the augmented prompt
    logger.info("Step 4: Calling LLM ...")
    llm_response = _call_llm(user_input, context)

    # Step 5 — Return the structured response
    if llm_response:
        logger.info("✅ Returning LLM-generated response.")
        return {
            "explanation":    llm_response.get("explanation", ""),
            "fix":            llm_response.get("fix", ""),
            "corrected_code": llm_response.get("corrected_code", ""),
        }
    else:
        # No LLM available — return best match from RAG retrieval
        logger.info("✅ Returning best match from RAG retrieval (no LLM).")
        return _extract_best_match(results, user_input)
