import requests

def ask_llm(context, query):
    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        },
    )

    return response.json()["response"]