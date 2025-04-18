import os
import gradio as gr # type: ignore

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint

# Path to FAISS vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"


# Load vectorstore (cached once)
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Set up custom prompt
def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Load HuggingFace model
def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",  # ✅ Fixed: specify task
        temperature=0.5,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": "512"
        }
    )

# Create QA chain
def create_qa_chain():
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HuggingFace token not found. Set HF_TOKEN environment variable.")

    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

    custom_prompt_template = """
    Use the pieces of information provided in the context to answer user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Don't provide anything outside of the given context.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """

    llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)
    vectorstore = get_vectorstore()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)}
    )

    return qa_chain

# Instantiate QA chain once
qa_chain = create_qa_chain()

# Chatbot logic
def chatbot(message, history):
    try:
        response = qa_chain.invoke({"query": message})
        answer = response["result"]
        sources = response["source_documents"]

        # Format source references
        source_texts = "\n".join([doc.metadata.get("source", "Unknown source") for doc in sources])
        formatted_response = f"{answer}\n\n**Sources:**\n{source_texts}"

        return formatted_response
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio Chat UI
chat_ui = gr.ChatInterface(
    fn=chatbot,
    title="Ask Chatbot!",
    chatbot=gr.Chatbot(height=500),
    theme="soft"
)

# Launch the Gradio app
if __name__ == "__main__":
    chat_ui.launch()
