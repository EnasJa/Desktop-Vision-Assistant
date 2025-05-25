import os
from dotenv import load_dotenv
import logging
import platform

# Load environment variables
load_dotenv()

# System details
SYSTEM = platform.system()  # 'Darwin' for macOS, 'Windows' for Windows
IS_MAC = SYSTEM == 'Darwin'
IS_WINDOWS = SYSTEM == 'Windows'

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TEMP_DIR = os.path.join(BASE_DIR, 'temp')

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# OpenAI API (for speech recognition)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Get from environment variable

# Speech recognition settings
RECORDING_DURATION = 5  # seconds
SAMPLE_RATE = 16000

# Vision settings
YOLO_MODEL = "yolov8n.pt"  # Nano model (smallest and fastest)
CONFIDENCE_THRESHOLD = 0.5
MAX_DETECTIONS = 20

# NLP settings
NLP_MODEL = "gpt2"  # Default model, can be changed to other Hugging Face models

# GUI settings
WINDOW_TITLE = "Desktop Vision Assistant"
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS = 30

# Logging
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"