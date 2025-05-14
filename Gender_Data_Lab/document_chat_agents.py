# Importing the necessary libraries
import openai
import pandas as pd
import numpy as np

client = openai

# Accessing the openai api_key
client.api_key = "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"
# The new workflow for ai in the gender data lab--------------------------------------------


# The function for the language detection and translation:
def detect_and_translate_to_english(text):
    system_prompt = """You are a translation assistant. Detect the language of the user input and translate it to English.
Always return a JSON with:
- "detected_language": The language name, e.g., "Kinyarwanda", "French"
- "translated_text": The English translation of the input"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )

    import json
    return json.loads(response['choices'][0]['message']['content'])



# The function for translating back to the user input language:
def translate_to_user_language(text, target_language):
    system_prompt = f"""Translate the following text into {target_language}. Keep Markdown formatting if present. Do not explain or add anything."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )

    return response['choices'][0]['message']['content']



# The query enhancer function:
def enhance_query(user_input):
    system_prompt = f"""You are an AI assistant that helps break down user queries into enhanced, well-structured queries.
You must always return a JSON object with two keys:
- "enhanced_query": A detailed, clarified, and expanded version of the user's input. Include tasks to do, conditions, show greetings or emotions if applicable, and propose tables/figures if useful.
- "retrievals_required": A boolean value — true if the query needs document-based content from external sources, false if it's just a general/greeting/emotional question.

Your enhanced query will be passed to another assistant that specializes in answering questions about **gender-related topics in Rwanda** on an information-sharing platform. Do NOT answer the question yourself — just enhance and classify it.

Examples:
User: Hello there!
→ "enhanced_query": "The user greeted the assistant with 'Hello there!'. Respond warmly.", "retrievals_required": false

User: What are the statistics of women in agriculture in Rwanda for 2020?
→ "enhanced_query": "Provide a detailed summary of statistics related to women's participation in agriculture in Rwanda in 2020. Include figures if available.", "retrievals_required": true

### User query:
{user_input}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt}
        ],
        temperature=0.3
    )

    response = response['choices'][0]['message']['content']
    import json
    return json.loads(response)




# The function for responding if the query is likely to require documents:
def respond_with_retrieval(enhanced_query, retrieved_rows):
    """
    This function generates a response using GPT-4o-mini based on the enhanced query
    and the retrieved documents (rows from your vector DB).
    """
    
    # Prepare a summary of retrieved contents (usually 3–5 top texts)
    combined_context = "\n\n".join(retrieved_rows['text'].dropna())
    
    system_prompt = f"""You are a highly knowledgeable assistant on gender-related topics in Rwanda. 
Use the provided documents to answer the user's query accurately and in detail.

- Always cite the content when used ("According to the document...")
- Keep your language clear, professional, and easy to follow
- Include statistics, facts, or key findings where relevant
- If the documents contain tabular data, format it using Markdown-style tables in your answer to improve clarity.
- If an image is available in the documents (via Markdown), include it in your answer using the same Markdown format
- If the documents are not enough to fully answer the query, politely say so, but still summarize what is known

Only answer based on the provided documents.

---

### 📄 documents
{combined_context}

---

### ❓ user's query
**{enhanced_query}**

---
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": system_prompt}
        ],
        temperature=0,
        max_tokens=1000
    )

    return response['choices'][0]['message']['content']




# The function for responding if the query is not likely to require documents:
def respond_without_retrieval(enhanced_query):
    """
    Responds to the user's query when no retrieval is required,
    e.g., greetings, emotional responses, or casual conversation.
    """

    system_prompt = f"""You are a friendly, helpful assistant working on a platform that shares gender-related knowledge about Rwanda.

Your job is to respond to non-technical questions such as:
- Greetings and emotional check-ins ("Hello", "I'm sad today", "Thank you")
- Casual or polite interactions ("You're helpful", "Goodbye", "What do you think?")
- Simple general questions not requiring document support

Guidelines:
- Be warm, respectful, and human-like.
- Do not reference external documents.
- Be concise but caring.
- If appropriate, guide the user toward asking a technical gender-related question.

Always format your response clearly in Markdown."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": enhanced_query}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Use "gpt-4o-mini" if that's your alias
        messages=messages,
        temperature=0.7,
        max_tokens=400
    )

    return response.choices[0].message["content"]




# The function for embeddings generation:
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=text, model=model)
    return response['data'][0]['embedding']

# The function for calculating similarity:
def calculate_similarity(query_embedding, db_embeddings):
    similarities = np.dot(db_embeddings, query_embedding)
    return similarities


# The function for semantic searching:
def query_system(question, df, model="text-embedding-ada-002"):
    query_embedding = get_embedding(question, model=model)
    db_embeddings = np.vstack(df['embeddings'])
    similarities = calculate_similarity(query_embedding, db_embeddings)
    top_4_indices = np.argsort(similarities)[-4:][::-1]
    retrieved_rows = df.iloc[top_4_indices]
    return retrieved_rows