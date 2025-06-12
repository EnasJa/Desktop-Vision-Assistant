```markdown
# Desktop Vision Assistant


A sophisticated AI-powered desktop application that combines speech recognition, computer vision, and natural language processing to create an intelligent visual assistant.

## Features

-  Real-time speech recognition for voice commands
-  Computer vision with object detection (using YOLOv8)
-  Natural language processing for command understanding
-  Desktop interaction capabilities
-  Live camera feed with object detection overlay
-  Multi-modal interaction (voice + visual)

## Installation

### Prerequisites
- Python 3.8+
- pip
- Webcam
- Microphone

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/EnasJa/Desktop-Vision-Assistant.git
   cd desktop-vision-assistant
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Usage

1. Run the application:
   ```bash
   python main.py
   ```

2. Basic commands:
   - "What do you see?" - Get description of current view
   - "Find [object]" - Search for specific objects in view
   - "Stop listening" - Deactivate voice recognition
   - "Start camera" - Enable camera feed

3. Interface controls:
   - Start/Stop camera feed
   - Toggle object detection
   - Adjust detection confidence threshold

## Project Structure

```
desktop-vision-assistant/
├── models/               # Model files
├── modules/              # Core application modules
│   ├── __init__.py
│   ├── gui.py            # Graphical user interface
│   ├── nlp_processor.py  # Natural language processing
│   ├── speech_recognition.py  # Voice command handling
│   ├── utils.py          # Utility functions
│   └── vision_processing.py  # Computer vision processing
├── temp/                 # Temporary files
├── venv/                 # Virtual environment
├── .env                  # Environment variables
├── .gitignore
├── config.py             # Configuration settings
├── main.py               # Main application entry point
├── README.md
├── requirements.txt      # Python dependencies
└── yolov8n.pt            # YOLOv8 model weights
```

## Configuration

Edit `config.py` to adjust:
- Camera settings
- Speech recognition parameters
- Object detection confidence thresholds
- UI preferences


## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV for computer vision
- SpeechRecognition library
- All contributors and open-source libraries used
```

---

*This project was developed as part of the Artificial Intelligence in Robotics course at Shamoon College of Engineering (SCE).*

---