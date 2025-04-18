import os
from dotenv import load_dotenv
load_dotenv()

import gradio as gr # type: ignore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint


# Your own modules
from brain_of_doctor import encode_image, analyze_image_with_query
from voice_of_patient import transcribe_with_groq
from voice_of_doctor import text_to_speech_with_elevenlabs


# ========== VECTOR DB SETUP ==========
DB_FAISS_PATH = "vectorstore/db_faiss"

def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

def create_qa_chain():
    HF_TOKEN = os.environ.get("HF_TOKEN")
    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

    custom_prompt_template = """
    Use the pieces of information provided in the context to answer user's question.
    If you don't know the answer, just say that you don't know.
    Don't provide anything outside of the context.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk.
    """

    llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)
    vectorstore = get_vectorstore()

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)}
    )

qa_chain = create_qa_chain()

# ========== CHATBOT FUNCTION ==========
def chatbot(message, history):
    try:
        response = qa_chain.invoke({"query": message})
        answer = response["result"]
        sources = response["source_documents"]
        source_texts = "\n".join([doc.metadata.get("source", "Unknown source") for doc in sources])
        return f"{answer}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ========== VOICE + IMAGE DIAGNOSIS ==========
system_prompt = """You have to act as a professional doctor, I know you are not but this is for learning purposes.
What's in this image? Do you find anything wrong with it medically?
If you make a differential, suggest some remedies for them. Do not add any numbers or special characters in your response.
Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Do not say 'In the image I see' but say 'With what I see, I think you have ....'
Don't respond as an AI model in markdown. Your answer should mimic that of an actual doctor, not an AI bot.
Keep your answer concise (max 2 sentences). No preamble, start your answer right away, please."""

def diagnose(audio_path, image_path):
    if not audio_path:
        return "No audio provided.", "No diagnosis generated.", None

    transcript = transcribe_with_groq(
        stt_model="whisper-large-v3",
        audio_filepath=audio_path,
        GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
    )

    if image_path:
        encoded_img = encode_image(image_path)
        full_prompt = system_prompt + " " + transcript
        diagnosis = analyze_image_with_query(
            query=full_prompt,
            encoded_image=encoded_img,
            model="llama-3-vision-alpha"
        )
    else:
        diagnosis = "No image provided."

    output_audio_path = "final_response.mp3"
    text_to_speech_with_elevenlabs(diagnosis, output_audio_path)

    return transcript, diagnosis, output_audio_path

# ========== GRADIO UI ==========
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 AI Doctor With: Chat, Voice & Vision")

    with gr.Tab("💬 Chat with AI Doctor"):
        chatbot_ui = gr.ChatInterface(
            fn=chatbot,
            title="Ask Anything (based on docs)",
            chatbot=gr.Chatbot(height=400),
            theme="soft"
        )

    with gr.Tab("🩻 Voice & Vision Diagnosis"):
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Speak Symptoms")
                image_input = gr.Image(type="filepath", label="🖼️ Upload Face Image")
                diagnose_btn = gr.Button("🩺 Diagnose Me")
            with gr.Column():
                transcript_box = gr.Textbox(label="📝 Transcription", lines=2)
                diagnosis_box = gr.Textbox(label="🧾 Diagnosis", lines=3)
                audio_output = gr.Audio(label="🔊 AI Doctor's Voice")

        diagnose_btn.click(
            fn=diagnose,
            inputs=[audio_input, image_input],
            outputs=[transcript_box, diagnosis_box, audio_output]
        )

    gr.Markdown("---\n👨‍⚕️ Powered by LangChain, LLaMA-3, ElevenLabs, and Whisper.")

demo.launch(debug=True)
