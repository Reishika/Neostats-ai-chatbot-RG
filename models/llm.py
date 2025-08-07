# LLM.py
import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = AzureChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version="2025-01-01-preview"
)
import json
def classify_response_and_relevance(bot_response: str, user_query: str) -> dict:
    prompt = f"""
You are a strict classifier.

Given the chatbot's response and the user's original query, perform the following:

1. **Classify the bot response** as either:
   - **"positive"**: The response is helpful, informative, complete, and contains **no uncertain, incomplete, or negative language**.
   - **"negative"**: The response is vague with negative intend, says "I don't know", "not found", "no information", or anything indicating **lack of knowledge, inability to help, or misalignment with the query**.

   Examples of negative indicators include:
   - "I do not know"
   - "This information is not available"
   - "I am not sure"
   - "Not found in the knowledge base"
   - "not explicitly mentioned"
   - "Cannot answer"
   - "The context provided does not include any information "

2. **Classify whether the user query is related to banking or insurance.**
   - Return "yes" if the query is about banking, finance, loans, accounts, cards, claims, insurance, premiums, policies, etc.
   - Return "no" if it's unrelated.

👉 **Only respond with a compact JSON object. Do not use markdown, code blocks, or explanations.**

Required format:
{{
    "response_class": "...",
    "is_relevant": "..."
}}

Bot Response:
\"\"\"{bot_response}\"\"\"

User Query:
\"\"\"{user_query}\"\"\"
"""
    ...

    try:
        response = llm.invoke([
            {"role": "system", "content": "You are a helpful assistant trained to classify responses and queries."},
            {"role": "user", "content": prompt}
        ])

        content = response.content.strip()

        # In case LLM wrapped response in triple quotes or backticks
        content = content.replace("```json", "").replace("```", "").strip()

        classification = json.loads(content)
        print(f"Classification result: {classification}")
        return classification

    except Exception as e:
        print("Error in classification:", e)
        return {"response_class": "positive", "is_relevant": "no"}  # safe fallback
