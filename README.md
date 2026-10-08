# 📺 YouTube RAG Assistant

A fast, Retrieval-Augmented Generation (RAG) pipeline built with Streamlit that allows you to ask questions about any YouTube video. The application fetches the video's transcript, splits it, embeds it into a vector store, and answers your queries using either blazing-fast cloud LLMs via **Groq** or local models via **Ollama**.

## ✨ Features
- **YouTube Transcript Extraction:** Automatically pulls transcripts from YouTube URLs.
- **Vector Search (FAISS):** Embeds and indexes video transcripts efficiently using HuggingFace embeddings.
- **Multiple Model Providers:**
  - **Groq API:** For ultra-fast cloud inference (e.g., Llama 3).
  - **Local Ollama:** For privacy-focused, entirely local inference.
- **Interactive UI:** Built with Streamlit for a clean, user-friendly experience.

## 🛠️ Prerequisites
- Python 3.8+
- [Ollama](https://ollama.com/) (Optional: Only if you intend to run local models)
- A [Groq API Key](https://console.groq.com/keys) (Optional: Only if you intend to use Groq)

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/YouTube-RAG-Assistant.git
   cd YouTube-RAG-Assistant
   ```

2. **Install the dependencies:**
   It is highly recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

## 🧠 How to Use
1. Open the app in your browser (typically at `http://localhost:8501`).
2. In the left sidebar, choose your **Model Provider**:
   - **Groq:** You will be prompted to paste your Groq API Key.
   - **Local Ollama:** Provide the name of the local model you want to use (e.g., `llama3`). Make sure Ollama is running in the background.
3. Paste a **YouTube Video URL** into the main input field and click **Process Video**.
4. Once the transcript is processed, you can ask any question related to the video, and the RAG assistant will provide context-aware answers!

## ☁️ Deployment (Streamlit Community Cloud)
You can easily deploy this application for free using Streamlit Community Cloud:
1. Push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub and select the repository.
4. Set the main file path as `app.py` and click **Deploy**.
> **Note:** Local Ollama models cannot be executed when deployed to the cloud. You must use the Groq provider on the deployed version.

## 📄 License
This project is open-source and available under the MIT License.
