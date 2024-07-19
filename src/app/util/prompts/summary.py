from langchain_core.prompts import ChatPromptTemplate

messages = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are generating concise and accurate summaries based on the information found in the text.",
        ),
        ("human", "Generate a summary of the following input: {question}\nSummary:"),
    ]
)
