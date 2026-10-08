import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Add a nice UI
st.set_page_config(page_title="YouTube RAG Assistant", layout="centered")

st.title("📺 YouTube RAG Assistant")
st.markdown("Ask questions about any YouTube video using Groq or local Ollama models.")
st.markdown("*Showcasing a fast retrieval-augmented generation pipeline.*")

st.sidebar.title("Configuration")
provider = st.sidebar.selectbox("Select Model Provider", ["Groq", "Local Ollama"])

groq_api_key = None
ollama_model = None

if provider == "Groq":
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
    st.sidebar.markdown("[Get a free Groq API key here](https://console.groq.com/keys)")
elif provider == "Local Ollama":
    ollama_model = st.sidebar.text_input("Ollama Model Name", value="llama3")
    st.sidebar.markdown("Make sure Ollama is running locally. *Note: Local models can't be used when deployed to the cloud.*")

# Initialize session state for vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

video_url = st.text_input("Enter YouTube Video URL:")
if st.button("Process Video"):
    if video_url:
        with st.spinner("Fetching transcript and building index..."):
            try:
                # Extract video ID
                if "v=" in video_url:
                    video_id = video_url.split("v=")[1].split("&")[0]
                elif "youtu.be/" in video_url:
                    video_id = video_url.split("youtu.be/")[1].split("?")[0]
                else:
                    video_id = video_url
                
                # 1. Document Ingestion
                ytt = YouTubeTranscriptApi()
                transcript_list = ytt.list_transcripts(video_id) if hasattr(ytt, 'list_transcripts') else ytt.list(video_id)
                transcript = transcript_list.find_transcript(["en"])
                fetched = transcript.fetch()
                
                # Depending on the version, fetched items might be dicts or objects
                if len(fetched) > 0 and isinstance(fetched[0], dict):
                    full_text = " ".join(item['text'] for item in fetched)
                else:
                    full_text = " ".join(item.text for item in fetched)
                
                # 2. Splitting
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_text(full_text)
                documents = [Document(page_content=chunk) for chunk in chunks]
                
                # 3. Embedding and Vector Store
                embeddings = HuggingFaceEmbeddings()
                vector_store = FAISS.from_documents(documents, embeddings)
                
                st.session_state.vector_store = vector_store
                st.success("✅ Video processed successfully! You can now ask questions.")
            except Exception as e:
                st.error(f"Error processing video: {e}")
    else:
        st.warning("Please enter a valid YouTube URL.")

question = st.text_input("Ask a question about the video:")
if st.button("Get Answer"):
    if question and st.session_state.vector_store:
        with st.spinner("Generating answer..."):
            try:
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={'k': 3})
                retrieved_docs = retriever.invoke(question)
                context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
                
                prompt = PromptTemplate(
                    template="""
                      You are a helpful assistant.
                      Answer ONLY from the provided transcript context.
                      If the context is insufficient, just say you don't know.

                      {context}
                      Question: {question}
                    """,
                    input_variables=['context', 'question']
                )
                
                final_prompt = prompt.invoke({"context": context_text, "question": question})
                
                if provider == "Groq":
                    if not groq_api_key:
                        st.error("Please enter a Groq API Key in the sidebar.")
                        st.stop()
                    llm = ChatGroq(
                        groq_api_key=groq_api_key,
                        model_name="llama-3.1-8b-instant",   
                        temperature=0.2
                    )
                elif provider == "Local Ollama":
                    from langchain_community.llms import Ollama
                    llm = Ollama(
                        model=ollama_model,
                        temperature=0.2
                    )
                
                answer = llm.invoke(final_prompt.text)
                st.markdown("### Answer:")
                st.write(answer.content)
            except Exception as e:
                st.error(f"Error generating answer: {e}.")
    elif not st.session_state.vector_store:
        st.warning("Please process a video first.")
    else:
        st.warning("Please enter a question.")
