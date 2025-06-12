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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """
    Function to handle audio recording using Streamlit's built-in audio recorder.
    This function is a placeholder as actual recording is handled by Streamlit's UI.
    
    Args:
    file_path (str): Path to save the recorded audio file.
    timeout (int): Maximum time to wait for a phrase to start (in seconds).
    phrase_time_limit (int): Maximum time for the phrase to be recorded (in seconds).
    """
    # This function is now just a placeholder as recording is handled by Streamlit's UI
    logging.info("Audio recording is handled by Streamlit's UI")
    return file_path

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
