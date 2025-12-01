"""
MCP server that provides tools to save and search memories using OpenAI's vector stores.
"""

import os
import tempfile
import sys
from dotenv import load_dotenv, find_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# Load environment variables from .env file (if present)
_ = load_dotenv(find_dotenv())

# Configuration: allow override via environment for easier debugging
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))

# Initialize OpenAI client
client = OpenAI()

# Name of the vector store used for memories
VECTOR_STORE_NAME = "MEMORIES_STORE"

# Initialize FastMCP instance
mcp = FastMCP("Memories")


def get_or_create_vector_store():
    """Retrieve the vector store by name or create it if it doesn't exist."""
    try:
        stores = client.vector_stores.list()
    except Exception as exc:  # broad here to surface helpful messages
        print(f"Error listing vector stores: {exc}", file=sys.stderr)
        raise

    for store in stores:
        if getattr(store, "name", None) == VECTOR_STORE_NAME:
            return store

    try:
        return client.vector_stores.create(name=VECTOR_STORE_NAME)
    except Exception as exc:
        print(f"Error creating vector store '{VECTOR_STORE_NAME}': {exc}", file=sys.stderr)
        raise


@mcp.tool()
def save_memory(memory: str) -> dict:
    """Save a memory string to the vector store."""
    vector_store = get_or_create_vector_store()
    # Write memory to a temporary file and upload it
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            tmp_path = f.name
            f.write(memory)
            f.flush()

        # Upload the file using a context manager to ensure closing the file
        with open(tmp_path, "rb") as fileobj:
            client.vector_stores.files.upload_and_poll(
                vector_store_id=vector_store.id,
                file=fileobj,
            )
    except Exception as exc:
        print(f"Error saving memory: {exc}", file=sys.stderr)
        return {"status": "error", "error": str(exc)}
    finally:
        # Best-effort cleanup of the temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return {"status": "saved", "vector_store_id": vector_store.id}


@mcp.tool()
def search_memory(query: str) -> dict:
    """Search memories in the vector store using the Assistants API."""
    vector_store = get_or_create_vector_store()

    try:
        # Create a temporary assistant with file_search enabled
        assistant = client.beta.assistants.create(
            name="Memory Search Assistant",
            instructions="You are a helpful assistant that searches through stored memories.",
            model="gpt-4o-mini",
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
        )

        # Create a thread with the search query
        thread = client.beta.threads.create(
            messages=[{"role": "user", "content": f"Search for: {query}"}]
        )

        # Run the assistant
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id
        )

        # Get the messages
        messages = client.beta.threads.messages.list(thread_id=thread.id)

        # Extract the assistant's response
        results = []
        for message in messages.data:
            if message.role == "assistant":
                for content in message.content:
                    if hasattr(content, 'text') and hasattr(content.text, 'value'):
                        results.append(content.text.value)

        # Clean up: delete the temporary assistant and thread
        client.beta.assistants.delete(assistant.id)
        client.beta.threads.delete(thread.id)

        return {"status": "ok", "results": results}

    except Exception as exc:
        print(f"Error searching vector store: {exc}", file=sys.stderr)
        return {"status": "error", "error": str(exc), "results": []}


if __name__ == "__main__":
    # When running via stdio (e.g., from client.py), don't print to stdout as it interferes with JSONRPC
    # All logging should go to stderr
    print(f"Starting MCP memory server (vector store: {VECTOR_STORE_NAME})", file=sys.stderr)
    try:
        # Run in stdio mode (default) for MCP client communication
        # For MCP Inspector, run separately with: mcp.run(host=MCP_HOST, port=MCP_PORT)
        mcp.run()
    except OSError as exc:
        print(f"Failed to start MCP server: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error running MCP: {exc}", file=sys.stderr)
        sys.exit(1)