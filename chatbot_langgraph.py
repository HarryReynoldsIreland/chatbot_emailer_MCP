"""Basic chatbot — LangGraph with Gmail MCP tool integration.

A local Gmail MCP server (gmail_mcp_server.py) is launched as a stdio
subprocess at startup. Its send_email tool is bound to the model so
the bot can email the conversation when asked.
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
USER_EMAIL = os.getenv("USER_EMAIL", "")
SYSTEM_PROMPT = (
    "You are a friendly, concise assistant. Keep replies short unless asked. "
    f"The user's email address is {USER_EMAIL}. "
    "When the user asks you to email them the conversation, call the "
    "send_email tool with that address, a clear subject, and a readable "
    "transcript of this chat as the body."
)

SERVER_PATH = str(Path(__file__).parent / "gmail_mcp_server.py")


async def build_chatbot():
    client = MultiServerMCPClient(
        {
            "gmail": {
                "command": sys.executable,
                "args": [SERVER_PATH],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    model = ChatAnthropic(model=MODEL, temperature=0.7).bind_tools(tools)

    def call_model(state: MessagesState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [model.invoke(messages)]}

    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")
    return builder.compile(checkpointer=MemorySaver())


async def main() -> None:
    chatbot = await build_chatbot()
    config = {"configurable": {"thread_id": "cli-thread"}}
    print("LangGraph chatbot with Gmail — type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue
        result = await chatbot.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        print(f"Bot: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    asyncio.run(main())
