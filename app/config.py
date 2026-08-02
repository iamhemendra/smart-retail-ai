"""
Central configuration and a lightweight API-key auth dependency.

For real deployment, replace API_KEY with an env var (see .env.example) and
consider OAuth2/JWT instead of a static key if this ever handles real
customer data.
"""
import os
from fastapi import Header, HTTPException, status

API_KEY = os.getenv("SMART_RETAIL_API_KEY", "dev-key-change-me")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def require_api_key(x_api_key: str = Header(default=None)) -> None:
    """FastAPI dependency: raises 401 if X-API-Key header doesn't match."""
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key. Send it via the X-API-Key header.",
        )
