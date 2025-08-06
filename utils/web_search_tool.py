from config.config import client, openai_client  # Azure OpenAI client assumed configured here
from langchain_core.tools import tool

@tool("Web Search", description="Tool for performing web searches using the Tavily API and returning either concise or detailed responses.")
def answer_with_web_search(query: str, mode: str) -> str:
    """
    Executes a web search using Tavily and returns a concise or detailed response using Azure OpenAI.

    Args:
        query (str): Search term.
        mode (str): 'concise' or 'detailed'.

    Returns:
        str: Final formatted response.
    """
    try:
        print(f"🔍 Query: {query}")
        print(f"📌 Mode: {mode}")

        # Step 1: Tavily search
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=True
        )

        # Get top result content
        results = response.get("results", []) if isinstance(response, dict) else getattr(response, "results", [])
        if not results:
            return "No results found from web search."

        content = results[0].get("content", "") if isinstance(results[0], dict) else getattr(results[0], "content", "")
        if not content:
            return "The search result did not return valid content."

        # Step 2: Use Azure OpenAI to transform result
        if mode.lower() == "concise":
            prompt = f"Summarize the following content in a concise 2-3 sentence answer:\n\n{content}"
        else:
            prompt = f"Expand and elaborate the following content in more detail, aiming for clarity and completeness:\n\n{content}"

        completion = openai_client.chat.completions.create(
            model="gpt-35-turbo",  # Replace with your actual Azure deployment name
            messages=[
                {"role": "system", "content": "You are a helpful assistant that reformats web search content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        final_answer = completion.choices[0].message.content.strip()
        print(f"✅ Transformed ({mode}) response generated.")
        return final_answer

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return f"Error: {str(e)}"
