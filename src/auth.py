import os
import hashlib
from pymongo import MongoClient
from src.config import MONGO_URI, MONGO_DB_NAME


def hash_password(password: str) -> str:
    """Return a SHA-256 hex digest for the given password string."""
    if not isinstance(password, str):
        password = str(password or "")
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_active_password():
    """
    Retrieves the active app password hash.
    Priority 1: Password hash stored in MongoDB (runtime override).
    Priority 2: Hash of APP_PASSWORD from environment (.env) as default.
    """
    # Fallback from environment (hash at runtime)
    default_password = os.getenv("APP_PASSWORD", "admin123")
    default_hash = hash_password(default_password)

    # If no Mongo URI configured, skip DB override
    if not MONGO_URI:
        return default_hash

    try:
        # Check MongoDB for an override
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        config_col = db["app_config"]

        # Look for a document with _id="access_password"
        override = config_col.find_one({"_id": "access_password"})

        client.close()

        if override and "value" in override and isinstance(override["value"], str):
            # Stored value is expected to be a hash
            return override["value"]
    except Exception as e:
        # Avoid breaking the app on DB issues; fall back to env
        print(f"Auth DB Error: {e}")

    return default_hash
