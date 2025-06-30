import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client
from typing import List, Dict, Any
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Supabase credentials
SUPABASE_URL: str = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
SUPABASE_TABLE_NAME: str = "n8n_workflows" # Match the table name you created

# OpenAI credentials
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

def get_openai_embedding(text: str) -> List[float]:
    """
    Generates an embedding for the given text using OpenAI's API.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        input=text,
        model=OPENAI_EMBEDDING_MODEL
    )
    return response.data[0].embedding

def embed_and_ingest_data():
    """
    Loads prepared workflow data, generates embeddings, and ingests into Supabase.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: Supabase URL or Key not found in .env file.")
        print("Please set SUPABASE_URL and SUPABASE_KEY.")
        return

    if not OPENAI_API_KEY:
        print("Error: OpenAI API Key not found in .env file.")
        print("Please set OPENAI_API_KEY.")
        return

    try:
        # Initialize Supabase client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Connected to Supabase.")
    except Exception as e:
        print(f"Error connecting to Supabase: {e}")
        return

    # Load prepared data
    prepared_data_path = "prepared_rag_data.json"
    if not os.path.exists(prepared_data_path):
        print(f"Error: Prepared data file '{prepared_data_path}' not found.")
        print("Please run prepare_rag_data.py first.")
        return

    with open(prepared_data_path, 'r', encoding='utf-8') as f:
        workflows_data: List[Dict[str, Any]] = json.load(f)

    print(f"Loaded {len(workflows_data)} workflows from {prepared_data_path}.")
    print(f"Generating embeddings and inserting into table: {SUPABASE_TABLE_NAME}")

    # Iterate and ingest
    for workflow in tqdm(workflows_data, desc="Ingesting workflows"):
        try:
            content_to_embed = workflow["content_for_vectorization"]
            embedding = get_openai_embedding(content_to_embed)

            # Prepare data for insertion (match your Supabase table schema)
            # Ensure the column names exactly match your Supabase table
            insert_data = {
                "filename": workflow["filename"],
                "workflow_name": workflow.get("workflow_name"),
                "description": workflow.get("description"),
                "trigger_type": workflow.get("trigger_type"),
                "complexity": workflow.get("complexity"),
                "node_count": workflow.get("node_count"),
                "integrations": workflow.get("integrations"), # Supabase will handle JSONB type
                "tags": workflow.get("tags"), # Supabase will handle JSONB type
                "mermaid_diagram": workflow.get("mermaid_diagram"),
                "raw_json": workflow.get("raw_json"), # Supabase will handle JSONB type
                "embedding": embedding # This is the vector
            }
            
            # Upsert (insert or update) based on filename to handle re-indexing
            # Assuming 'filename' column has a unique constraint or is used for upserting
            # Supabase's upsert typically requires the primary key, or a unique column.
            # We'll use filename as the unique identifier for upsert.
            response = supabase.table(SUPABASE_TABLE_NAME).upsert(insert_data, on_conflict="filename").execute()
            
            if response.data:
                pass # Successfully inserted/updated
            else:
                print(f"Warning: No data returned for {workflow['filename']}. Response: {response.status_code} {response.json()}")

        except Exception as e:
            print(f"Error processing {workflow.get('filename', 'unknown')}: {e}")
            continue

    print("\nData ingestion complete!")

if __name__ == "__main__":
    embed_and_ingest_data() 