""# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "faker",
#     "httpx",
#     "numpy",
#     "pillow",
#     "python-dateutil",
# ]
# ///
import hashlib
import httpx
import json
import logging
import numpy as np
import os
import re
import subprocess
from dateutil.parser import parse
from datagen import (
    get_markdown,
    get_dates,
    get_contacts,
    get_logs,
    get_docs,
    get_email,
    get_credit_card,
    get_comments,
    get_tickets,
)

openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
openai_api_key = os.getenv("OPENAI_API_KEY")

def num(string: str):
    return int(hashlib.sha256(string.encode()).hexdigest(), 16) % (2**64)  # Reduced collision risk

def mismatch(msg, expected, result):
    logging.error(f"\U0001F534 {msg}\n⚠️ EXPECTED:\n{expected}\n⚠️ RESULT:\n{result}")
    return False

async def run(task: str):
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            logging.warning(f"\U0001F7E1 Running task: {task.strip()}")
            response = await client.post("http://localhost:8000/run", params={"task": task})
            response.raise_for_status()
            response_text = json.dumps(response.json(), indent=2)
            logging.info(f"\U0001F7E2 HTTP {response.status_code} {response_text}")
            return response.status_code, response_text
        except httpx.HTTPError as e:
            logging.error(f"\U0001F534 HTTP request failed: {e}")
            return None, None

async def read(path: str):
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.get(f"http://localhost:8000/read?path={path}")
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as e:
            logging.error(f"\U0001F534 Cannot read {path}: {e}")
            return None

async def a2(email: str, file: str = "/data/format.md", **kwargs):
    original = get_markdown(email)
    try:
        expected = subprocess.run(
            ["npx", "prettier@3.4.2", "--stdin-filepath", file],
            input=original,
            capture_output=True,
            text=True,
            check=True,
            shell=False,
        ).stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"\U0001F534 Prettier execution failed: {e}")
        return False
    
    await run(f"Format the contents of `{file}` using `prettier@3.4.2`, updating the file in-place")
    result = await read(file)
    if not result or result.strip() != expected.strip():
        return mismatch(file, expected, result)
    return True

async def a9(email, **kwargs):
    data = get_comments(email)
    if len(data) < 2:
        logging.warning("\U0001F7E1 Not enough comments for similarity calculation.")
        return False

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                f"{openai_api_base}/embeddings",
                headers={"Authorization": f"Bearer {openai_api_key}"},
                json={"model": "text-embedding-3-small", "input": data},
            )
            response.raise_for_status()
            embeddings = np.array([emb["embedding"] for emb in response.json()["data"]])
        except (httpx.HTTPError, KeyError) as e:
            logging.error(f"\U0001F534 Failed to retrieve embeddings: {e}")
            return False

    similarity = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(similarity, -np.inf)
    i, j = np.unravel_index(similarity.argmax(), similarity.shape)
    expected = "\n".join(sorted([data[i], data[j]]))
    
    await run("Find the most similar pair of comments from `/data/comments.txt` and write them to `/data/comments-similar.txt`")
    result = await read("/data/comments-similar.txt")
    sorted_result = "\n".join(sorted([line for line in result.split("\n") if line.strip()]))
    if sorted_result != expected:
        return mismatch("/data/comments-similar.txt", expected, result)
    return True

if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate tasks with configurable logging")
    parser.add_argument("--email", default="user@example.com", help="Set the email address")
    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    parser.add_argument("--log-level", default="INFO", choices=levels, help="Set logging level")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(message)s\n")
    asyncio.run(main(args.email))
