#!/usr/bin/env python
import os
import sys
from datetime import datetime
import hashlib

from pymongo import MongoClient

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def hash_password(password: str) -> str:
    if not isinstance(password, str):
        password = str(password or "")
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def main():
    # Load .env
    if load_dotenv is not None:
        load_dotenv()

    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME", "jyotish_ai_cache")
    app_password = os.getenv("APP_PASSWORD")

    if not mongo_uri:
        print("Error: MONGO_URI is required in .env.")
        sys.exit(2)
    if not app_password:
        print("Error: APP_PASSWORD is required in .env.")
        sys.exit(2)

    password_hash = hash_password(app_password)

    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        col = db["app_config"]

        result = col.update_one(
            {"_id": "access_password"},
            {
                "$set": {
                    "value": password_hash,
                    "algo": "sha256",
                    "updatedAt": datetime.utcnow(),
                }
            },
            upsert=True,
        )

        client.close()

        if result.modified_count or result.upserted_id is not None:
            print(
                f"Password hash set successfully in '{db_name}.app_config' with _id='access_password'."
            )
            sys.exit(0)
        else:
            print("No changes made. Existing document retained.")
            sys.exit(0)

    except Exception as e:
        print(f"Failed to set password: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
