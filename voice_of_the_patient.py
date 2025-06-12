# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

#Step1: Setup Audio recorder (ffmpeg & portaudio)
# ffmpeg, portaudio, pyaudio
import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import os
from groq import Groq
import sounddevice as sd
import numpy as np
import wave
import streamlit as st
import tempfile
import soundfile as sf
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def record_audio(file_path):
    """
    Record audio using Streamlit's built-in audio recorder
    """
    try:
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        # Use Streamlit's audio recorder
        audio_bytes = st.audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            sample_rate=16000
        )

        if audio_bytes is not None:
            # Convert audio bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Save the audio using soundfile
            sf.write(temp_path, audio_array, 16000)
            
            # Move the temporary file to the desired location
            os.replace(temp_path, file_path)
            logger.info(f"Audio recorded and saved to {file_path}")
            return True
        else:
            logger.warning("No audio was recorded")
            return False

    except Exception as e:
        logger.error(f"Error recording audio: {str(e)}")
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return False

# Step2: Setup Speech to text–STT–model for transcription
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
stt_model="whisper-large-v3"

def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY):
    """
    Transcribe audio using Groq's API.
    
    Args:
    stt_model (str): The model to use for transcription.
    audio_filepath (str): Path to the audio file.
    GROQ_API_KEY (str): Groq API key.
    
    Returns:
    str: Transcribed text.
    """
    client = Groq(api_key=GROQ_API_KEY)
    
    if not os.path.exists(audio_filepath):
        raise ValueError("Audio file does not exist.")
        
    audio_file = open(audio_filepath, "rb")
    transcription = client.audio.transcriptions.create(
        model=stt_model,
        file=audio_file,
        language="en"
    )

    return transcription.text

def transcribe_audio(file_path):
    """
    Transcribe audio using Google Speech Recognition
    """
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return None

def main():
    st.title("Voice of the Patient")
    
    # Create a directory for audio files if it doesn't exist
    os.makedirs("recordings", exist_ok=True)
    
    # Record audio
    if st.button("Start Recording"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join("recordings", f"recording_{timestamp}.wav")
        
        if record_audio(file_path):
            st.success("Recording completed!")
            
            # Transcribe the audio
            text = transcribe_audio(file_path)
            if text:
                st.write("Transcription:", text)
            else:
                st.error("Failed to transcribe the audio")
        else:
            st.error("Failed to record audio")

if __name__ == "__main__":
    main()
