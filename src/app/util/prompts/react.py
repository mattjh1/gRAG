from langchain.tools.render import render_text_description_and_args
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.util.tools import tools

system_message = f"""Answer the following questions as best you can.
You can answer directly if the user is greeting you or similar.
Otherise, you have access to the following tools:

{render_text_description_and_args(tools).replace('{', '{{').replace('}', '}}')}

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use)
and a `action_input` key (with the input to the tool going here).
The only values that should be in the "action" field are: {[t.name for t in tools]}
The $JSON_BLOB should only contain a SINGLE action, 
do NOT return a list of multiple actions.
Here is an example of a valid $JSON_BLOB:
```
{{
    "action": $TOOL_NAME,
    "action_input": $INPUT
}}
```
The $JSON_BLOB must always be enclosed with triple backticks!

ALWAYS use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action:```
$JSON_BLOB
```
Observation: the result of the action... 
(this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Reminder to always use the exact characters `Final Answer` when responding.'
"""

messages = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            system_message,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
