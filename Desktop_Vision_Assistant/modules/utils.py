"""
Utility functions for the Desktop Vision Assistant
"""

import os
import sys
import time
import logging
import platform
import subprocess
import shutil
import requests
import cv2
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

def check_environment():
    """Check if the environment has all necessary dependencies installed"""
    logger.info("Checking environment")
    
    # Check Python version
    python_version = sys.version_info
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.warning("Python 3.8 or higher is recommended")
    
    # Check system
    system = platform.system()
    logger.info(f"Operating system: {system}")
    
    # Check for required executables
    required_executables = {
        "ffmpeg": check_ffmpeg()
    }
    
    for name, available in required_executables.items():
        if available:
            logger.info(f"{name} is available")
        else:
            logger.warning(f"{name} is not available or not in PATH")
    
    return all(required_executables.values())

def check_ffmpeg():
    """Check if ffmpeg is installed and available in PATH"""
    try:
        # Try to run ffmpeg -version
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

def download_file(url, destination, chunk_size=8192):
    """
    Download a file from a URL with progress bar
    
    Args:
        url (str): URL to download from
        destination (str): Path where the file will be saved
        chunk_size (int, optional): Size of chunks to download. Defaults to 8192.
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Make sure the destination directory exists
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        
        # Check if file already exists
        if os.path.exists(destination):
            logger.info(f"File already exists: {destination}")
            return True
        
        # Start download
        logger.info(f"Downloading {url} to {destination}")
        response = requests.get(url, stream=True)
        
        # Check if request was successful
        if response.status_code != 200:
            logger.error(f"Failed to download {url}, status code: {response.status_code}")
            return False
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Show progress bar
        progress_bar = tqdm(
            total=total_size, 
            unit='B', 
            unit_scale=True, 
            desc=os.path.basename(destination)
        )
        
        # Download and save file
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        logger.info(f"Download completed: {destination}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        # Clean up partial download
        if os.path.exists(destination):
            os.remove(destination)
        return False

def list_available_cameras():
    """
    List available camera devices
    
    Returns:
        list: List of available camera indices
    """
    logger.info("Listing available cameras")
    available_cameras = []
    
    # Check up to 5 camera indices
    for i in range(5):
        cap = None
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Camera is available
                available_cameras.append(i)
                logger.info(f"Camera {i} is available")
        except Exception as e:
            logger.debug(f"Error checking camera {i}: {e}")
        finally:
            if cap is not None:
                cap.release()
    
    return available_cameras

def test_microphone():
    """
    Test if microphone is working
    
    Returns:
        bool: True if microphone is working, False otherwise
    """
    try:
        import sounddevice as sd
        
        logger.info("Testing microphone")
        
        # Get default device info
        device_info = sd.query_devices(kind='input')
        logger.info(f"Default input device: {device_info['name']}")
        
        # Try recording a short sample
        duration = 1  # seconds
        sample_rate = 16000
        logger.info(f"Recording {duration} second test...")
        
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait for recording to finish
        
        # Check if recording contains sound
        if recording.max() > 0.01:
            logger.info("Microphone test successful")
            return True
        else:
            logger.warning("Microphone test detected no sound")
            return False
            
    except Exception as e:
        logger.error(f"Error testing microphone: {e}")
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run tests when this module is executed directly
    check_environment()
    test_microphone()