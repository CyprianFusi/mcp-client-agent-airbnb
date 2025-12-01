"""
Streamlit UI for an Airbnb chatbot using MCP and OpenAI.
"""

import streamlit as st
import asyncio
import json
import sys
import traceback
from dotenv import load_dotenv, find_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI


# Set the event loop policy to WindowsProactorEventLoopPolicy at the top of the file to fix subprocess support on Windows.
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Load environment variables
_= load_dotenv(find_dotenv())

# Define parameters for both MCP servers
server_params_1 = StdioServerParameters(
    command="npx",
    args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
)

server_params_2 = StdioServerParameters(
    command="uv",
    args=["--directory", "./", "run", "server.py"],
)

# Helper function to convert MCP tools to OpenAI tool format
def get_openai_tools(combined_tools):
    """Convert MCP tools to OpenAI tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            },
        }
        for tool in combined_tools
    ]

# Asynchronous function to handle chat responses with tool calls
async def chat_response(messages, session1, session2, tools_result_1, tools_result_2, openai_tools, client):
    """Handle chat response with potential tool calls from both servers."""

    # Initial LLM call
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
    )
    tool_calls = response.choices[0].message.tool_calls

    # Handle any tool calls
    if response.choices[0].message.tool_calls:
        messages.append(response.choices[0].message)
        for tool_execution in tool_calls:
            # Decide which session to use based on tool name
            if any(t.name == tool_execution.function.name for t in tools_result_1.tools):
                session = session1
            else:
                session = session2

            # Execute tool call
            result = await session.call_tool(
                tool_execution.function.name,
                arguments=json.loads(tool_execution.function.arguments),
            )
            # Add tool response to conversation
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_execution.id,
                    "content": result.content[0].text,
                }
            )

            # Get response from LLM
            response = client.chat.completions.create(
                model='gpt-4o',
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )
            if response.choices[0].finish_reason == "tool_calls":
                tool_calls.extend(response.choices[0].message.tool_calls)
            if response.choices[0].finish_reason == "stop":
                messages.append(response.choices[0].message)
                return response.choices[0].message.content
    else:
        messages.append(response.choices[0].message)
        return response.choices[0].message.content

# Synchronous wrapper for the asynchronous chat response function
def sync_chat_response(messages, user_input):
    """Synchronous wrapper for chat response with both servers."""
    async def _chat():
        """Asynchronous chat handler."""
        async with stdio_client(server_params_1) as (read1, write1), stdio_client(server_params_2) as (read2, write2):
            async with ClientSession(read1, write1) as session1, ClientSession(read2, write2) as session2:
                # Initialize both servers
                await session1.initialize()
                await session2.initialize()

                # Get tools from both servers
                tools_result_1 = await session1.list_tools()
                tools_result_2 = await session2.list_tools()

                # Combine tools
                combined_tools = tools_result_1.tools + tools_result_2.tools
                openai_tools = get_openai_tools(combined_tools)

                client = OpenAI()
                messages.append({"role": "user", "content": user_input})
                response = await chat_response(messages, session1, session2, tools_result_1, tools_result_2, openai_tools, client)
                return response
    return asyncio.run(_chat())

# Streamlit UI
def main():
    """Main function for Streamlit UI."""
    st.title("Airbnb Chatbot (MCP + OpenAI)")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Reset button
    if st.button("Reset Chat"):
        st.session_state["messages"] = []
        st.session_state["history"] = []
        st.rerun()

    # Display chat history
    for user_msg, bot_msg in st.session_state["history"]:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)

    # Chat input (this automatically clears after submission)
    user_input = st.chat_input("Ask me about Airbnb listings...")

    if user_input:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = sync_chat_response(st.session_state["messages"], user_input)
                    st.markdown(response)
                    # Save to history
                    st.session_state["history"].append((user_input, response))
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
                    st.error(error_msg)

# Entry point for Streamlit
def run_streamlit():
    main()

if __name__ == "__main__":
    run_streamlit()
