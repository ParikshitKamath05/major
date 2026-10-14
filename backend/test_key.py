"""
test_key.py  — Cohere key smoke test
Tests both cohere.chat() (as specified) and cohere.embed() (what the pipeline
actually uses for ChromaDB queries and semantic entropy).
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
print(f"Key in use: {COHERE_API_KEY[:12]}...{COHERE_API_KEY[-4:]}\n")

try:
    import cohere
except ImportError:
    print("FAILED  cohere package not installed")
    sys.exit(1)

client = cohere.Client(COHERE_API_KEY)

# ── TEST 1: Chat (as specified in Task 0) ─────────────────────────────────
print("=" * 55)
print("TEST 1: cohere.chat(model='command-r-plus', message='Say OK')")
print("=" * 55)
try:
    resp = client.chat(model="command-r-plus", message="Say OK")
    text = resp.text if hasattr(resp, "text") else str(resp)
    print(f"SUCCESS  | First 50 chars: {text[:50]!r}")
    chat_ok = True
except Exception as e:
    print(f"FAILED   | {type(e).__name__}")
    print(f"Error (first 1000 chars):\n{str(e)[:1000]}")
    chat_ok = False

# ── TEST 2: Embed (what the pipeline uses) ────────────────────────────────
print()
print("=" * 55)
print("TEST 2: cohere.embed(model='embed-english-light-v3.0')")
print("  (pipeline uses this for ChromaDB queries & semantic entropy)")
print("=" * 55)
try:
    embed_resp = client.embed(
        texts=["Hello world"],
        model="embed-english-light-v3.0",
        input_type="search_query",
    )
    vec = embed_resp.embeddings[0]
    print(f"SUCCESS  | Vector dim={len(vec)}, first 3 values: {vec[:3]}")
    embed_ok = True
except Exception as e:
    print(f"FAILED   | {type(e).__name__}")
    print(f"Error (first 1000 chars):\n{str(e)[:1000]}")
    embed_ok = False

# ── SUMMARY ───────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"  Chat  (command-r-plus):          {'OK' if chat_ok  else 'FAILED'}")
print(f"  Embed (embed-english-light-v3):  {'OK' if embed_ok else 'FAILED'}")

if chat_ok and embed_ok:
    print("\n  ✅ KEY IS VALID — proceed to Branch A (full eval)")
    sys.exit(0)
elif embed_ok and not chat_ok:
    print("\n  ⚠️  Embed OK but Chat failed — pipeline embeddings work,")
    print("     but check if chat quota is shared. May still proceed.")
    sys.exit(2)
else:
    print("\n  ❌ KEY FAILED — likely 429 quota exhausted → Branch B (Groq swap)")
    sys.exit(1)
