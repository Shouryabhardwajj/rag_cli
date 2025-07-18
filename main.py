import os
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
genai.configure(api_key=API_KEY)

def setup_rag_system(csv_path: str):
    print("Setting up the RAG system...")
    try:
        df = pd.read_csv(csv_path)
        df['term_definition'] = df['term'].astype(str) + ": " + df['explanation'].astype(str)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return None, None, None

    print("Loading embedding model...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    print("Creating embeddings for the knowledge base...")
    embeddings = encoder.encode(df['term_definition'].tolist(), show_progress_bar=True)
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))
    print("Setup complete!")
    return df, encoder, index

def retrieve_relevant_docs(query: str, encoder: SentenceTransformer, index: faiss.Index, df: pd.DataFrame, k: int = 3, relevance_threshold: float = 1.0):
    query_embedding = encoder.encode([query])
    distances, top_k_indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    if distances[0][0] > relevance_threshold:
        return [], False
    context_docs = [df.iloc[i]['term_definition'] for i in top_k_indices[0]]
    return context_docs, True

def generate_rag_response(query: str, context: list):
    prompt = f"""
    Based on the following context about similar technical terms:
    Context:
    - {'\n- '.join(context)}

    Now, answer the user's question clearly and concisely.
    User's Question: "What is {query}?"

    Provide your answer in the following format:
    1. Explanation: A plain English explanation of the term.
    2. Analogy: A simple, real-world analogy.
    3. Use Case: A short, practical use case.
    """
    print("\n--> Found relevant context. Generating response with RAG...")
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the response: {e}"

def main():
    data_df, encoder_model, faiss_index = setup_rag_system('tech_terms.csv')
    if data_df is None:
        return

    print("\nWelcome to the Tech Term Explainer CLI!")

    while True:
        user_query = input("\nEnter a technical term: ").strip()
        if user_query.lower() == 'exit':
            print("Have a nice day!")
            break
        if not user_query:
            continue

        retrieved_context, is_relevant = retrieve_relevant_docs(user_query, encoder_model, faiss_index, data_df)
        final_response = ""

        if is_relevant:
            final_response = generate_rag_response(user_query, retrieved_context)
        else:
            fallback_prompt = f"""
            Answer the user's question clearly and concisely.
            User's Question: "What is {user_query}?"

            Provide your answer in the following format:
            1. Explanation: A plain English explanation of the term.
            2. Analogy: A simple, real-world analogy.
            3. Use Case: A short, practical use case.
            """
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(fallback_prompt)
                final_response = response.text
            except Exception as e:
                final_response = f"An error occurred while generating the response: {e}"

        print("\n" + "="*50)
        print(final_response)
        print("="*50)

if __name__ == "__main__":
    main()