import asyncio
import os
import chromadb
import cohere
from dotenv import load_dotenv
from groq import AsyncGroq

from corrective import CorrectiveRetriever

load_dotenv()

# Adjust this path to match your main.py CHROMA_DIR
chroma_client = chromadb.PersistentClient(path="../chroma_db")
collection = chroma_client.get_collection(name="documents")

cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
async_groq = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))


def retrieve_fn(query: str, k: int):
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return (
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    )


async def main():
    retriever = CorrectiveRetriever(async_groq, cohere_client)
    result = await retriever.run(
        query="What is the role of the Prime Minister?",
        retrieve_fn=retrieve_fn,
    )
    print("\n=== RESULT ===")
    print(f"Method: {result.method}")
    print(f"Expanded queries: {result.expanded_queries}")
    print(f"Total candidates: {result.total_candidates}")
    print(f"Final chunks: {len(result.documents)}")
    print(f"Rerank scores: {result.rerank_scores}")


if __name__ == "__main__":
    asyncio.run(main())