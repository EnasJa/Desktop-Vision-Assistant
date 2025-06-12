import os
import time
import tempfile
import threading
import logging
import sounddevice as sd
import soundfile as sf
import numpy as np
from queue import Queue
from openai import OpenAI

from config import RECORDING_DURATION, SAMPLE_RATE, TEMP_DIR, OPENAI_API_KEY

logger = logging.getLogger(__name__)

class SpeechRecognizer:
    def __init__(self):
        logger.info(f"Initializing speech recognizer with OpenAI API")
        # Initialize OpenAI client
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.recording = False
        self.audio_queue = Queue()
        self.transcription_queue = Queue()
        self.recording_thread = None
        self.processing_thread = None
        logger.info(f"Speech recognizer initialized")
        
    def start(self):
        """Start the speech recognition system"""
        logger.info("Starting speech recognition system")
        self.recording = True
        self.recording_thread = threading.Thread(target=self._record_audio_loop)
        self.processing_thread = threading.Thread(target=self._process_audio_loop)
        self.recording_thread.daemon = True
        self.processing_thread.daemon = True
        self.recording_thread.start()
        self.processing_thread.start()
        
    def stop(self):
        """Stop the speech recognition system"""
        logger.info("Stopping speech recognition system")
        self.recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1.0)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
            
    def get_transcription(self):
        """Get the latest transcription if available"""
        if not self.transcription_queue.empty():
            return self.transcription_queue.get()
        return None
        
    def _record_audio_loop(self):
        """Continuously record audio in chunks"""
        logger.info(f"Recording audio with sample rate: {SAMPLE_RATE}")
        while self.recording:
            audio_data = self._record_audio(duration=RECORDING_DURATION)
            if audio_data is not None and len(audio_data) > 0:
                self.audio_queue.put(audio_data)
            time.sleep(0.1)  # Small delay to prevent CPU overload
            
    def _process_audio_loop(self):
        """Process recorded audio chunks"""
        logger.info("Starting audio processing loop")
        while self.recording:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                transcription = self._transcribe_audio(audio_data)
                if transcription and transcription.strip():  # Only queue non-empty transcriptions
                    logger.info(f"Transcription: {transcription}")
                    self.transcription_queue.put(transcription)
            time.sleep(0.1)  # Small delay to prevent CPU overload
            
    def _record_audio(self, duration=3):
        """Record audio for a specific duration with noise reduction"""
        try:
            logger.debug(f"Recording {duration} seconds of audio")
            
            # Record with higher quality settings
            audio_data = sd.rec(
                int(duration * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                device=None,  # Use default device
                blocking=True  # Ensure complete recording
            )
            
            # Simple noise gate - remove very quiet sounds
            threshold = 0.01
            audio_data[np.abs(audio_data) < threshold] = 0
            
            # Check if audio has meaningful content
            if np.max(np.abs(audio_data)) < 0.005:
                logger.debug("Audio too quiet, skipping")
                return None
                
            return audio_data
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            return None
            
    def _transcribe_audio(self, audio_data):
        """Transcribe audio data using OpenAI API with enhanced settings"""
        try:
            # Save audio to temporary file with higher quality
            temp_file = os.path.join(TEMP_DIR, f"recording_{int(time.time())}.wav")
            
            # Normalize audio to improve quality
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Save with higher quality settings
            sf.write(temp_file, audio_data, SAMPLE_RATE, subtype='PCM_16')
            
            # Transcribe with OpenAI API with enhanced parameters
            logger.debug(f"Transcribing audio file: {temp_file}")
            
            with open(temp_file, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",  # Specify language for better accuracy
                    prompt="This is a voice command for a desktop vision assistant.",  # Context helps
                    temperature=0.1  # Lower temperature for more consistent results
                )
            
            # Clean up temp file
            os.remove(temp_file)
            
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ""
            
    def transcribe_file(self, file_path):
        """Transcribe an existing audio file"""
        try:
            logger.info(f"Transcribing file: {file_path}")
            
            with open(file_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            return response.text
        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
            return ""


# Simple test function
def test_speech_recognition():
    recognizer = SpeechRecognizer()
    print("Recording for 5 seconds...")
    audio = recognizer._record_audio(5)
    if audio is not None:
        print("Transcribing...")
        text = recognizer._transcribe_audio(audio)
        print(f"Transcription: {text}")
    else:
        print("Failed to record audio")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Test the speech recognition
    test_speech_recognition()