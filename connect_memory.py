import os
from dotenv import load_dotenv, find_dotenv

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

# 🧠 Use the new preferred imports (optional but recommended for future-proofing)
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings

# 🔁 Load environment variables
load_dotenv(find_dotenv())

# ✅ Show token prefix (for debug, optional)
HF_TOKEN = os.environ.get("HF_TOKEN")
print("🔍 HF_TOKEN (first 10 chars):", HF_TOKEN[:10] if HF_TOKEN else "NOT FOUND")

# 🔧 Model config
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.5,
            task="text-generation",  # ✅ Set correct task
            model_kwargs={
                "max_length": 512
            }
        )
        return llm
    except Exception as e:
        print("❌ ERROR: Failed to load HuggingFace model -", e)
        exit(1)

# 💬 Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't make anything up.
Only answer using the context.

Context: {context}
Question: {question}

Start the answer directly. No small talk.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

# 📦 Load vector store
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# 🔗 Create Retrieval QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# 🚀 Query loop
try:
    user_query = input("Write Query Here: ")
    response = qa_chain.invoke({'query': user_query})
    print("\n✅ RESULT:\n", response["result"])
    print("\n📄 SOURCE DOCUMENTS:")
    for i, doc in enumerate(response["source_documents"]):
        print(f"\n--- Document {i+1} ---")
        print(doc.page_content)
except Exception as e:
    print("❌ ERROR during query execution:", e)
