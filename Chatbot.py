# ------------------------------------------------------------------------
# Streamlit Chat with History/Memory - Amazon Bedrock and LangChain
# ------------------------------------------------------------------------

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

import streamlit as st
# from langchain_openai import ChatOpenAI
# ------------------------------------------------------------------------


model_kwargs =  { 
    "max_tokens_to_sample": 2048,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

model_id = "anthropic.claude-instant-v1"

# ------------------------------------------------------------------------
# LangChain

template = [
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
]

prompt = ChatPromptTemplate.from_messages(template)

# model = BedrockChat(
#     client=bedrock_runtime,
#     model_id=model_id,
#     model_kwargs=model_kwargs,
# )



model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=st.secrets["openai_api_key"],  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)

# Chain without History
chain = prompt | model | StrOutputParser()

# Streamlit Chat Message History
history = StreamlitChatMessageHistory(key="chat_messages")

# Chain with History
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,  # Always return the instance created earlier
    input_messages_key="question",
    history_messages_key="history",
)

# ------------------------------------------------------------------------
# Streamlit



# Page title
st.set_page_config(page_title='Streamlit Chat')

# Clear Chat History fuction
def clear_chat_history():
    history.messages.clear()

with st.sidebar:
    st.title('Streamlit Chat')
    st.subheader('With Memory :brain:')
    streaming_on = st.toggle('Streaming')
    st.button('Clear Chat History', on_click=clear_chat_history)
    st.divider()
    st.write("History Logs")
    st.write(history.messages)

if history.messages == []:
    history.add_ai_message("How may I assist you today?")

for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

# Chat Input - User Prompt 
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)

    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    config = {"configurable": {"session_id": "any"}}

    if streaming_on:
        # Chain - Stream
        placeholder = st.empty()
        full_response = ''
        for chunk in chain_with_history.stream({"question": prompt}, config):
            full_response += chunk
            placeholder.chat_message("ai").write(full_response)
        placeholder.chat_message("ai").write(full_response)

    else:
        # Chain - Invoke
        response = chain_with_history.invoke({"question": prompt}, config)
        st.chat_message("ai").write(response)