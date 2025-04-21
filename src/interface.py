

import streamlit as st
from src.extracting_knowledge import generate_response


def load_interface():
    # Set page configuration
    st.set_page_config(page_title="Doctor's Mind", page_icon="🧑‍💻")
    st.title("🧑‍💻 Doctor's Mind: Smart Medical Assistant")

    # Initialize chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input box for user
    user_input = st.chat_input("Type your message...")

    # If user sends a message
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate bot response
        bot_response = generate_response(user_input)

        # Add bot message to history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response["result"])


