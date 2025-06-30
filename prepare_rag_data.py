import json
import os
from pathlib import Path
from typing import Dict, List, Any

# Assuming workflow_db and api_server are in the same parent directory
# Adjust imports if your project structure differs
from workflow_db import WorkflowDatabase
from api_server import generate_mermaid_diagram # Import the function directly

def _ensure_strings_in_list(data_list: List[Any]) -> List[str]:
    """
    Ensures all items in a list are strings, converting dicts/other types if possible.
    This helps prevent 'sequence item expected str, dict found' errors during joining.
    """
    cleaned_list = []
    for item in data_list:
        if isinstance(item, str):
            cleaned_list.append(item)
        elif isinstance(item, dict):
            # Try to get 'name' or 'id' from dict, otherwise convert to string
            if 'name' in item:
                cleaned_list.append(str(item['name']))
            elif 'id' in item:
                cleaned_list.append(str(item['id']))
            else:
                cleaned_list.append(str(item))
        else:
            cleaned_list.append(str(item))
    return cleaned_list

def prepare_workflow_for_rag(db_instance: WorkflowDatabase) -> List[Dict[str, Any]]:
    """
    Extracts relevant metadata and Mermaid diagram code for all workflows,
    preparing them for vectorization and RAG ingestion.
    """
    prepared_data = []
    workflows_dir = Path("workflows")

    if not workflows_dir.exists():
        print(f"Error: Workflows directory '{workflows_dir.absolute()}' not found.")
        return []

    json_files = list(workflows_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in '{workflows_dir.absolute()}' to process.")
        return []

    print(f"Preparing data for {len(json_files)} workflows...")

    for file_path in json_files:
        filename = file_path.name
        try:
            # 1. Get structured metadata from the database (or analyze directly if not indexed)
            # For this preparation script, directly using analyze_workflow_file is more robust
            # as it doesn't rely on the database being fully indexed with latest changes.
            workflow_meta = db_instance.analyze_workflow_file(str(file_path))

            if not workflow_meta:
                print(f"Skipping {filename}: Could not analyze workflow metadata.")
                continue

            # 2. Load raw JSON for Mermaid diagram generation
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_json_data = json.load(f)

            nodes = raw_json_data.get('nodes', [])
            connections = raw_json_data.get('connections', {})

            # 3. Generate Mermaid diagram code
            mermaid_diagram = generate_mermaid_diagram(nodes, connections)

            # 4. Prepare content for vectorization
            # Combine textual metadata and diagram for a rich vectorized document
            # Ensure integrations and tags are lists of strings before joining
            cleaned_integrations = _ensure_strings_in_list(workflow_meta.get('integrations', []))
            cleaned_tags = _ensure_strings_in_list(workflow_meta.get('tags', []))

            content_for_vectorization = f"""
            Workflow Name: {workflow_meta.get('name', 'N/A')}
            Filename: {workflow_meta.get('filename', 'N/A')}
            Description: {workflow_meta.get('description', 'N/A')}
            Trigger Type: {workflow_meta.get('trigger_type', 'N/A')}
            Complexity: {workflow_meta.get('complexity', 'N/A')}
            Node Count: {workflow_meta.get('node_count', 0)}
            Integrations: {', '.join(cleaned_integrations)}
            Tags: {', '.join(cleaned_tags)}

            Mermaid Diagram:
            ```mermaid
            {mermaid_diagram}
            ```
            """
            
            # Remove excessive whitespace/newlines for cleaner embedding
            content_for_vectorization = ' '.join(content_for_vectorization.split()).strip()

            prepared_data.append({
                "filename": filename, # Unique ID for Supabase
                "workflow_name": workflow_meta.get('name'),
                "trigger_type": workflow_meta.get('trigger_type'),
                "complexity": workflow_meta.get('complexity'),
                "node_count": workflow_meta.get('node_count'),
                "integrations": cleaned_integrations,
                "tags": cleaned_tags,
                "description": workflow_meta.get('description'),
                "mermaid_diagram": mermaid_diagram, # Store raw diagram for display
                "content_for_vectorization": content_for_vectorization,
                "raw_json": raw_json_data # Include raw JSON if needed for full context after retrieval
            })

        except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
            print(f"Error processing {filename}: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred for {filename}: {e}")
            continue

    print(f"Finished preparing data for {len(prepared_data)} workflows.")
    return prepared_data

if __name__ == "__main__":
    # Ensure the database directory exists, as WorkflowDatabase will try to create it
    os.makedirs('database', exist_ok=True)
    db = WorkflowDatabase()
    
    # Optional: ensure workflows are indexed for the analyze_workflow_file to be accurate
    # db.index_all_workflows() # Uncomment if you want to ensure DB is up-to-date before analysis

    prepared_workflows = prepare_workflow_for_rag(db)

    # You can now save this data to a JSON file, or pass it to the next step
    # for embedding and Supabase ingestion.
    output_path = Path("prepared_rag_data.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prepared_workflows, f, indent=2, ensure_ascii=False)
    
    print(f"Prepared data saved to {output_path.absolute()}")
    print(f"Total entries ready for vectorization: {len(prepared_workflows)}") 