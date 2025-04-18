# AI Doctor - Final Working Version
import io
import os
import gradio as gr
from gtts import gTTS
import numpy as np
from PIL import Image
from brain_of_the_doctor import encode_image, analyze_image_with_query, generate_prescription
from voice_of_the_patient import record_audio, transcribe_with_groq

# Import custom avatar
from custom_avatar import SpeakingAvatar
avatar = SpeakingAvatar("portrait-3d-female-doctor[1].jpg")

# Supported languages mapping
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr"
}

def text_to_speech_bytes(text, language='en'):
    """Convert text to speech bytes with proper error handling"""
    try:
        lang_code = LANGUAGE_CODES.get(language, 'en')
        tts = gTTS(text=text, lang=lang_code)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        return audio_bytes.getvalue()
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        return None

SYSTEM_PROMPTS = {
    "English": "You are a professional doctor...",
    "Hindi": "आप एक पेशेवर डॉक्टर हैं...",
    "Marathi": "तुम्ही एक व्यावसायिक डॉक्टर आहात..."
}

# Cache for API responses
response_cache = {}

def process_inputs(audio, text, image, language, progress=gr.Progress()):
    try:
        progress(0.1, desc="Processing...")
        cache_key = f"{audio or text}{image}{language}"
        if cache_key in response_cache:
            return response_cache[cache_key]
        
        # Input handling
        if audio:
            input_text = transcribe_with_groq(
                stt_model="whisper-large-v3",
                audio_filepath=audio,
                GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
            )
        else:
            input_text = text
        
        # Image analysis
        if image:
            progress(0.3, desc="Analyzing image...")
            encoded_image = encode_image(image)
            diagnosis = analyze_image_with_query(
                query=SYSTEM_PROMPTS[language],
                encoded_image=encoded_image
            )
        else:
            diagnosis = f"Response to: {input_text}"

        # Generate audio in memory
        audio_bytes = text_to_speech_bytes(diagnosis, language)
        if not audio_bytes:
            raise ValueError("Failed to generate audio response")
            
        # Generate prescription
        prescription = generate_prescription(diagnosis, language)
            
        result = (input_text, diagnosis, (16000, np.frombuffer(audio_bytes, dtype=np.int16)), prescription,
                 gr.DownloadButton(visible=True))
        response_cache[cache_key] = result
        return result
        
    except Exception as e:
        return (f"Error: {str(e)}",) * 4 + (gr.DownloadButton(visible=False),)


# Load custom CSS
with open("medical_style.css", "r") as f:
    custom_css = f.read()

with gr.Blocks(title="VAIDYA - Medical Diagnosis", css=custom_css) as app:
    gr.Markdown("""
    # 🩺 AI Doctor - Medical Diagnosis System
    *Professional medical diagnosis powered by AI*
    """)
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("🎤 Voice Input"):
                    audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record your symptoms")
                with gr.TabItem("✍️ Text Input"):
                    text_input = gr.Textbox(label="Describe your symptoms", lines=3)
            
            with gr.Accordion("🖼️ Upload Medical Image (Optional)", open=False):
                image_input = gr.Image(type="filepath", label="Medical Image")
            
            with gr.Row():
                language = gr.Dropdown(
                    choices=list(LANGUAGE_CODES.keys()),
                    value="English",
                    label="Response Language",
                    scale=2
                )
                submit_btn = gr.Button("🔍 Get Diagnosis", variant="primary", scale=1)
        
        with gr.Column(scale=1):
            avatar_output = gr.Image(
                avatar.get_avatar(), 
                label="Your Doctor", 
                height=400, 
                width=300,
                show_label=True,
                elem_classes="doctor-avatar"
            )
    
    # Output Section
    with gr.Column(elem_classes="output-section"):
        gr.Markdown("## 📋 Diagnosis Results")
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(label="Your Input Summary", interactive=False)
                with gr.Accordion("🩺 Detailed Diagnosis", open=True):
                    diagnosis = gr.Textbox(label="", lines=5, interactive=False, elem_classes="diagnosis-box")
                audio_output = gr.Audio(label="🎧 Audio Diagnosis", interactive=False)
            with gr.Column(scale=1):
                with gr.Accordion("💊 Prescription", open=True):
                    prescription = gr.Textbox(label="", lines=10, interactive=True, elem_classes="prescription-box")
                    download_btn = gr.DownloadButton("📥 Download Prescription", visible=False)

    submit_btn.click(
        fn=process_inputs,
        inputs=[audio_input, text_input, image_input, language],
        outputs=[input_text, diagnosis, audio_output, prescription, download_btn]
    ).then(
        lambda: gr.Image(avatar.get_avatar()),
        outputs=[avatar_output]
    )

    download_btn.click(
        lambda text: (text, "prescription.txt"),
        inputs=[prescription],
        outputs=[download_btn]
    )

if __name__ == "__main__":
    try:
        print("Attempting to launch with public sharing...")
        app.launch(share=True)  # Let Gradio choose an available port
    except Exception as e:
        print(f"Failed to create public link: {str(e)}")
        print("Falling back to local URL only")
        try:
            app.launch()  # Try without port specification
        except Exception as e:
            print(f"Failed to launch application: {str(e)}")
            print("Please check if another instance is running or try a different port")
