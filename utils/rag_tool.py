# rag_tool.py

import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
INDEX_NAME = "index_field_1"

# Azure Search client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# Azure OpenAI client
llm_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)
TEMPLATES = {
    "concise": PromptTemplate.from_template("""
You are a helpful AI assistant. Always include the policy names. Using only the following context, answer the user's question briefly and clearly.

Context:
{context}

Question: {question}
Answer (concise):"""),
    "detailed": PromptTemplate.from_template("""
You are a helpful AI assistant. Always include the policy names. Using only the following context, answer the user's question in a detailed and informative manner.

Context:
{context}

Question: {question}
Answer (detailed):""")
}

def get_relevant_chunks(query: str, k: int = 5) -> list:
    """Search Azure Cognitive Search for top-k relevant documents with the policy names."""
    results = search_client.search(
        search_text=query,
        top=k,
        include_total_count=False
    )

    matched_chunks = []
    for doc in results:
        content = doc["content"]
        label = label_chunk_type(content)
        policy_name = doc["source"] if "source" in doc else "Unnamed Policy"
        matched_chunks.append({
            "content": content,
            "label": label,
            "policy": policy_name
        })
    return matched_chunks


def label_chunk_type(text: str) -> str:
    """Label the chunk based on its likely content."""
    text = text.lower()
    if "who can avail" in text or "eligibility" in text or "available for" in text or "age between" in text or "coverage" in text:
        return "eligibility"
    elif "premium" in text or "monthly premium" in text:
        return "premium"
    return "general"




def answer_with_knowledge_base(query: str, mode: str = "concise") -> str:
    """Answer a query using the vector store (Azure Cognitive Search) and Azure OpenAI."""

    try:
        chunks = get_relevant_chunks(query, k=5)
        if not chunks:
            return "I don't know based on the knowledge base."


        if any(word in query.lower() for word in ["age", "eligible", "avail", "child", "adult", "senior"]):
            eligible_chunks = [c for c in chunks if c["label"] == "eligibility"]
            if eligible_chunks:
                chunks_to_use = eligible_chunks
            else:
                chunks_to_use = chunks
        else:
            chunks_to_use = chunks

        #context = "\n\n".join(chunks[:5])

        context = ""
        used_policies = set()

        for c in chunks_to_use[:5]:
            context += f"Policy: {c['policy']}\nContent: {c['content']}\n\n"
            used_policies.add(c['policy'])
        prompt = TEMPLATES[mode].format(context=context, question=query)

        completion = llm_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"Error during RAG answering: {e}"
