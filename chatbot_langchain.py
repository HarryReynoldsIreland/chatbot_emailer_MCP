"""Basic chatbot — LangChain (LCEL) approach.

Memory: RunnableWithMessageHistory wraps a chain and looks up the
conversation by session_id from an in-process dict of histories.
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
SYSTEM_PROMPT = "You are a friendly, concise assistant. Keep replies short unless asked."

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

model = ChatAnthropic(model=MODEL, temperature=0.7)
chain = prompt | model

_store: dict[str, BaseChatMessageHistory] = {}


def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]


chatbot = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)


def main() -> None:
    session_id = "cli-session"
    config = {"configurable": {"session_id": session_id}}
    print("LangChain chatbot — type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue
        response = chatbot.invoke({"input": user_input}, config=config)
        print(f"Bot: {response.content}\n")


if __name__ == "__main__":
    main()
