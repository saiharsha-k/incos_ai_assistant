# chatbot_app.py

import streamlit as st
from coursework_assistant import RAGSystem

@st.cache_resource
def load_rag_system():
    return RAGSystem(
        model_path="Sai-Harsha-k/fine-tuned-model-t5-small",  # Replace with your actual model path on Hugging Face
        faiss_index_path="E:\Sai Harsha\From_23\MSc Data Science\DeepLearning_Applications\coursework_assistant\code\faiss_index.bin",
        embeddings_path="E:\Sai Harsha\From_23\MSc Data Science\DeepLearning_Applications\coursework_assistant\code\corpus_embeddings.json",
        corpus_path="E:\Sai Harsha\From_23\MSc Data Science\DeepLearning_Applications\coursework_assistant\code\processed_data.json"
    )

def main():
    st.title("Coursework Assistant Chatbot")

    rag = load_rag_system()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What would you like to know about the coursework?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = rag.answer_question(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()