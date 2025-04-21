# Doctor's Mind

Doctor's Mind is an intelligent medical assistant designed to help doctors make informed decisions by leveraging advanced natural language processing (NLP) models and a robust knowledge base.

The application uses **Streamlit** for the user interface, **Hugging Face** for language models, and **Pinecone** for vector-based retrieval.

---

## Features

- **Interactive Chat Interface**: A user-friendly chat interface built with Streamlit.
- **Medical Knowledge Retrieval**: Retrieves relevant medical knowledge from a vector store.
- **Customizable Prompt**: Tailored prompts ensure accurate and context-aware responses.
- **Hugging Face Integration**: Utilizes Hugging Face's `Mistral-7B-Instruct` model for generating responses.
- **Pinecone Vector Store**: Efficiently retrieves relevant documents using Pinecone.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/doctors-mind.git
cd doctors-mind
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory and add the following:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_pinecone_index_name
```

### 5. Run the Application

```bash
streamlit run app.py
```

---

## File Structure

```bash
.
├── app.py
├── requirements.txt
├── .env
├── README.md
├── utils/
│   ├── retrieval.py
│   └── prompts.py
├── assets/
│   └── logo.png
└── models/
    └── llm.py
```

---

## How It Works

1. **User Input**: The user types a query into the chat interface.
2. **Knowledge Retrieval**: The query is processed using Pinecone to retrieve relevant documents.
3. **Response Generation**: The Hugging Face model generates a response based on the retrieved documents and a custom prompt.
4. **Interactive Chat**: The response is displayed in the Streamlit chat interface.

---

## Technologies Used

- **Streamlit**: For building the interactive user interface.
- **Hugging Face**: For language model integration (Mistral-7B-Instruct).
- **Pinecone**: For efficient vector-based document retrieval.
- **LangChain**: For chaining LLMs with retrieval mechanisms.
- **Python**: Core programming language.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [Hugging Face](https://huggingface.co/)
- [Pinecone](https://www.pinecone.io/)
- [LangChain](https://www.langchain.dev/)

---

**Doctor's Mind** - Empowering doctors with intelligent decision-making assistance.
