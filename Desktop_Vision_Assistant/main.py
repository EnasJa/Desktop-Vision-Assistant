#!/usr/bin/env python3
"""
Desktop Vision Assistant
A camera and microphone-based vision assistant for desktop computers

This project combines:
- Speech recognition (OpenAI API)
- Natural language processing (GPT or Hugging Face models)
- Computer vision (YOLOv8)
- GUI interface for user interaction
"""

import os
import sys
import logging
import colorlog
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config import (
    LOG_LEVEL, LOG_FORMAT, 
    YOLO_MODEL, NLP_MODEL
)

# Import modules
from modules.speech_recognition import SpeechRecognizer
from modules.vision_processing import VisionProcessor
from modules.nlp_processor import NLPProcessor
from modules.gui import run_gui

# Set up logger
def setup_logging(log_level=LOG_LEVEL):
    """Set up logging with colors"""
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)
    
    return root_logger

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Desktop Vision Assistant')
    
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index to use (default: 0)')
    
    parser.add_argument('--openai-api-key', type=str,
                        help='OpenAI API key (if not set in environment)')
    
    parser.add_argument('--yolo-model', type=str, default=YOLO_MODEL,
                        help=f'YOLOv8 model to use (default: {YOLO_MODEL})')
    
    parser.add_argument('--nlp-model', type=str, default=NLP_MODEL,
                        help=f'NLP model to use (default: {NLP_MODEL})')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    return parser.parse_args()

def main():
    """Main entry point for the application"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else LOG_LEVEL
    logger = setup_logging(log_level)
    logger.info("Starting Desktop Vision Assistant")
    
    # Set OpenAI API key from arguments if provided
    if args.openai_api_key:
        import os
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    
    try:
        # Check if API key is set
        from config import OPENAI_API_KEY
        if not OPENAI_API_KEY:
            logger.error("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable or use --openai-api-key")
            return 1
        
        # Create speech recognizer
        logger.info("Initializing Speech Recognizer with OpenAI API")
        speech_recognizer = SpeechRecognizer()
        
        # Create vision processor
        logger.info(f"Initializing Vision Processor with model: {args.yolo_model}")
        vision_processor = VisionProcessor()
        
        # Create NLP processor
        logger.info(f"Initializing NLP Processor with model: {args.nlp_model}")
        nlp_processor = NLPProcessor()
        
        # Start GUI
        logger.info("Starting GUI")
        run_gui(speech_recognizer, vision_processor, nlp_processor)
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Error running application: {e}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    # If running as script, execute main function
    sys.exit(main())