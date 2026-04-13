"""
endee_client.py — Endee Vector Database Client
================================================
Handles all interactions with the Endee vector database:
  • Creating the index for error-solution embeddings
  • Upserting vectors (error embeddings + metadata)
  • Querying for similar errors via cosine similarity search

Includes a graceful fallback mode when Endee server is not available,
ensuring the application works for demos and development.
"""

import logging
import math
from typing import List, Dict, Any, Optional

# ──────────────────────────────────────────────
#  Logger
# ──────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Try importing the Endee SDK
# ──────────────────────────────────────────────
try:
    from endee import Endee, Precision
    ENDEE_SDK_AVAILABLE = True
except ImportError:
    logger.warning("⚠️  Endee SDK not installed. Running in FALLBACK mode.")
    logger.warning("   Install with: pip install endee")
    ENDEE_SDK_AVAILABLE = False
    Endee = None
    Precision = None


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────
INDEX_NAME = "debug_errors"          # Name of our Endee index
VECTOR_DIMENSION = 384               # all-MiniLM-L6-v2 output dimension
SPACE_TYPE = "cosine"                # Similarity metric
DEFAULT_TOP_K = 3                    # How many similar results to retrieve
SEARCH_EF = 128                      # HNSW ef parameter for search accuracy


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors (fallback math)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EndeeClient:
    """
    Wrapper around the Endee Python SDK.
    Provides simple methods to seed data and perform similarity search.

    Falls back to in-memory vector search when Endee server is unavailable,
    so the app always works.
    """

    def __init__(self, base_url: str = "http://localhost:8080/api/v1", auth_token: Optional[str] = None):
        """
        Initialize the Endee client.

        Args:
            base_url:   The Endee server API endpoint.
            auth_token: Optional authentication token.
        """
        self.base_url = base_url
        self.client = None
        self.index = None
        self.connected = False

        # In-memory fallback storage
        self._fallback_records: List[Dict[str, Any]] = []

        if ENDEE_SDK_AVAILABLE:
            try:
                self.client = Endee(auth_token or "")
                self.client.set_base_url(base_url)
                logger.info("Endee client initialised (base_url=%s)", base_url)
            except Exception as e:
                logger.warning("Could not initialize Endee client: %s", e)
                self.client = None
        else:
            logger.info("Endee SDK not available — using in-memory fallback.")

    # ──────────────────────────────────────────
    #  Index Management
    # ──────────────────────────────────────────
    def create_index_if_not_exists(self) -> None:
        """
        Create the vector index for storing error embeddings.
        Skips creation if it already exists.
        Falls back gracefully if Endee is unreachable.
        """
        if not self.client:
            logger.info("Endee not available — using in-memory fallback index.")
            return

        try:
            # Try to get the existing index first
            self.index = self.client.get_index(name=INDEX_NAME)
            self.connected = True
            logger.info("Index '%s' already exists — reusing.", INDEX_NAME)
        except Exception:
            try:
                # Index doesn't exist — create it
                logger.info("Creating Endee index '%s' (dim=%d, space=%s)...",
                            INDEX_NAME, VECTOR_DIMENSION, SPACE_TYPE)
                self.client.create_index(
                    name=INDEX_NAME,
                    dimension=VECTOR_DIMENSION,
                    space_type=SPACE_TYPE,
                    precision=Precision.FLOAT32
                )
                self.index = self.client.get_index(name=INDEX_NAME)
                self.connected = True
                logger.info("Index '%s' created successfully.", INDEX_NAME)
            except Exception as e:
                logger.warning("Could not create/access Endee index: %s", e)
                logger.info("Falling back to in-memory vector search.")
                self.connected = False

    # ──────────────────────────────────────────
    #  Upsert Vectors
    # ──────────────────────────────────────────
    def upsert_vectors(self, records: List[Dict[str, Any]]) -> None:
        """
        Insert or update vectors in the Endee index.
        Falls back to in-memory storage if Endee is unavailable.
        """
        # Always store in fallback (used if Endee search fails too)
        self._fallback_records = records

        if not self.connected or self.index is None:
            logger.info("Stored %d vectors in local fallback memory.", len(records))
            return

        try:
            # Endee allows max 1000 vectors per upsert call
            batch_size = 1000
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                self.index.upsert(batch)
                logger.info("Upserted batch %d–%d (%d vectors) to Endee.",
                            i, i + len(batch) - 1, len(batch))
        except Exception as e:
            logger.warning("Endee upsert failed: %s — data stored in fallback.", e)

    # ──────────────────────────────────────────
    #  Similarity Search
    # ──────────────────────────────────────────
    def search(self, query_vector: List[float], top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Search the index for the closest error-solution pairs.
        Falls back to in-memory cosine similarity if Endee is unavailable.

        Args:
            query_vector: The embedding of the user's error/code input.
            top_k:        Number of results to return.

        Returns:
            A list of dicts, each containing:
                { "id", "similarity", "meta" }
        """
        # Try Endee first
        if self.connected and self.index is not None:
            try:
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    ef=SEARCH_EF,
                    include_vectors=False
                )
                logger.info("Endee search returned %d result(s).", len(results))
                return results
            except Exception as e:
                logger.warning("Endee search failed: %s — using fallback.", e)

        # Fallback: in-memory cosine similarity search
        return self._fallback_search(query_vector, top_k)

    def _fallback_search(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        In-memory cosine similarity search over stored records.
        Used when Endee server is not available.
        """
        if not self._fallback_records:
            logger.warning("No fallback records available — returning empty.")
            return []

        # Calculate similarity for each record
        scored = []
        for record in self._fallback_records:
            sim = _cosine_similarity(query_vector, record["vector"])
            scored.append({
                "id": record["id"],
                "similarity": sim,
                "meta": record.get("meta", {}),
            })

        # Sort by similarity descending
        scored.sort(key=lambda x: x["similarity"], reverse=True)

        results = scored[:top_k]
        logger.info("Fallback search returned %d result(s) (top similarity: %.3f).",
                     len(results), results[0]["similarity"] if results else 0)
        return results

    @property
    def is_connected(self) -> bool:
        """Whether we have a live connection to Endee."""
        return self.connected
