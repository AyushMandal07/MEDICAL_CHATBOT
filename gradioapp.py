from dotenv import load_dotenv
load_dotenv()

import os
import gradio as gr # type: ignore
from brain_of_doctor import encode_image, analyze_image_with_query
from voice_of_patient import record_audio, transcribe_with_groq
from voice_of_doctor import text_to_speech_with_elevenlabs

system_prompt = """You have to act as a professional doctor, I know you are not but this is for learning purposes. 
What's in this image? Do you find anything wrong with it medically? 
If you make a differential, suggest some remedies for them. Do not add any numbers or special characters in your response. 
Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Do not say 'In the image I see' but say 'With what I see, I think you have ....' 
Don't respond as an AI model in markdown. Your answer should mimic that of an actual doctor, not an AI bot. 
Keep your answer concise (max 2 sentences). No preamble, start your answer right away, please."""

def process_inputs(audio_filepath, image_filepath):
    if not audio_filepath:
        return "No audio provided.", "No response generated.", None

    speech_to_text_output = transcribe_with_groq(
        stt_model="whisper-large-v3",
        audio_filepath=audio_filepath,
        GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
    )

    if image_filepath:
        encoded_img = encode_image(image_filepath)
        full_prompt = system_prompt + " " + speech_to_text_output
        doctor_response = analyze_image_with_query(
            query=full_prompt,
            encoded_image=encoded_img,
            model="llama-3.2-11b-vision-preview"
        )
    else:
        doctor_response = "No image provided for me to analyze."

    output_audio_path = "final_response.mp3"
    text_to_speech_with_elevenlabs(input_text=doctor_response, output_filepath=output_audio_path)

    return speech_to_text_output, doctor_response, output_audio_path

with gr.Blocks(theme=gr.themes.Soft(), css="""
    .gr-button {
        background-color: #4f46e5 !important;
        color: white !important;
        border-radius: 1rem;
        padding: 0.75rem 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .gr-button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 15px rgba(79,70,229,0.3);
    }
    .gr-textbox, .gr-audio, .gr-image {
        border-radius: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    body {
        background: linear-gradient(to right, #f8fafc, #e0f7fa);
    }
""") as demo:
    gr.Markdown("""
        # 🧠 AI Doctor with Vision and Voice
        Speak your symptoms and upload a facial image. Let the AI doctor analyze and give you a smart diagnosis.
    """)
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Speak Your Symptoms")
            image_input = gr.Image(type="filepath", label="🖼️ Upload Face Image")
            submit_btn = gr.Button("🩺 Diagnose Me")

        with gr.Column():
            transcript_output = gr.Textbox(label="📝 Transcription", lines=2)
            diagnosis_output = gr.Textbox(label="🧾 Doctor's Response", lines=3)
            audio_output = gr.Audio(label="🔊 AI Doctor Speaks")

    submit_btn.click(
        fn=process_inputs,
        inputs=[audio_input, image_input],
        outputs=[transcript_output, diagnosis_output, audio_output]
    )

    gr.Markdown("""
        ---
        👨‍⚕️ Powered by multimodal LLMs + real-time voice & vision.
    """)

demo.launch(debug=True)