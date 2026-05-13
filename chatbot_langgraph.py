"""Basic chatbot — LangGraph approach.

Memory: a StateGraph holding a MessagesState (auto-appending list of
messages) is compiled with a MemorySaver checkpointer. Each invoke is
keyed by thread_id, which loads/saves the conversation transparently.
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
SYSTEM_PROMPT = "You are a friendly, concise assistant. Keep replies short unless asked."

model = ChatAnthropic(model=MODEL, temperature=0.7)


def call_model(state: MessagesState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = model.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_edge(START, "model")

chatbot = builder.compile(checkpointer=MemorySaver())


def main() -> None:
    thread_id = "cli-thread"
    config = {"configurable": {"thread_id": thread_id}}
    print("LangGraph chatbot — type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        print(f"Bot: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
