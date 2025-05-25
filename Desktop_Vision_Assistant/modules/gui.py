import os
import sys
import time
import logging
import threading
from queue import Queue
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QComboBox, QSlider, QCheckBox
)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QThread, QSize

from config import (
    WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT,
    CAMERA_WIDTH, CAMERA_HEIGHT, FPS
)

logger = logging.getLogger(__name__)

class VideoThread(QThread):
    """Thread to capture video frames"""
    frame_signal = pyqtSignal(np.ndarray)
    detection_signal = pyqtSignal(list)
    
    def __init__(self, vision_processor):
        super().__init__()
        self.vision_processor = vision_processor
        self.running = False
        
    def run(self):
        """Main thread loop for capturing frames"""
        self.running = True
        
        while self.running:
            # Get the latest frame and detection results
            frame = self.vision_processor.get_frame()
            detections = self.vision_processor.get_detection_results()
            
            if frame is not None:
                # Emit frame signal
                self.frame_signal.emit(frame)
                
            if detections is not None:
                # Emit detections signal
                self.detection_signal.emit(detections)
                
            # Sleep to match FPS
            time.sleep(1.0 / FPS)
            
    def stop(self):
        """Stop the thread"""
        self.running = False
        self.wait()


class DesktopVisionGUI(QMainWindow):
    """Main GUI window for the Desktop Vision Assistant"""
    
    def __init__(self, speech_recognizer, vision_processor, nlp_processor):
        super().__init__()
        
        # Store component references
        self.speech_recognizer = speech_recognizer
        self.vision_processor = vision_processor
        self.nlp_processor = nlp_processor
        
        # Initialize GUI
        self.init_ui()
        
        # Set up timer for checking transcriptions
        self.transcription_timer = QTimer(self)
        self.transcription_timer.timeout.connect(self.check_transcriptions)
        self.transcription_timer.start(100)  # Check every 100ms
        
        # Set up timer for checking NLP responses
        self.response_timer = QTimer(self)
        self.response_timer.timeout.connect(self.check_responses)
        self.response_timer.start(100)  # Check every 100ms
        
        # Set up video thread
        self.video_thread = VideoThread(self.vision_processor)
        self.video_thread.frame_signal.connect(self.update_frame)
        self.video_thread.detection_signal.connect(self.update_detections)
        
        # Start components
        self.start_components()
        
    def init_ui(self):
        """Initialize the user interface"""
        logger.info("Initializing GUI")
        
        # Set window properties
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel: Video feed and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(CAMERA_WIDTH, CAMERA_HEIGHT)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        left_layout.addWidget(self.video_label)
        
        # Camera controls
        camera_controls = QHBoxLayout()
        
        # Camera selection
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(100)
        self.camera_combo.addItem("Default Camera (0)")
        for i in range(1, 3):  # Add more camera options
            self.camera_combo.addItem(f"Camera {i}")
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        camera_controls.addWidget(QLabel("Camera:"))
        camera_controls.addWidget(self.camera_combo)
        
        # Start/Stop button
        self.camera_button = QPushButton("Stop Camera")
        self.camera_button.clicked.connect(self.toggle_camera)
        camera_controls.addWidget(self.camera_button)
        
        # Toggle detection checkbox
        self.detection_checkbox = QCheckBox("Show Detections")
        self.detection_checkbox.setChecked(True)
        camera_controls.addWidget(self.detection_checkbox)
        
        left_layout.addLayout(camera_controls)
        
        # Right panel: Command output and controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Transcription display
        right_layout.addWidget(QLabel("Recognized Speech:"))
        self.transcription_text = QTextEdit()
        self.transcription_text.setReadOnly(True)
        self.transcription_text.setMaximumHeight(100)
        right_layout.addWidget(self.transcription_text)
        
        # Command output
        right_layout.addWidget(QLabel("Assistant Response:"))
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.response_text.setMinimumHeight(200)
        right_layout.addWidget(self.response_text)
        
        # Detected objects list
        right_layout.addWidget(QLabel("Detected Objects:"))
        self.objects_text = QTextEdit()
        self.objects_text.setReadOnly(True)
        self.objects_text.setMaximumHeight(150)
        right_layout.addWidget(self.objects_text)
        
        # Audio controls
        audio_controls = QHBoxLayout()
        
        # Start/Stop audio button
        self.audio_button = QPushButton("Stop Listening")
        self.audio_button.clicked.connect(self.toggle_audio)
        audio_controls.addWidget(self.audio_button)
        
        # Test command input
        test_controls = QHBoxLayout()
        self.test_command_input = QTextEdit()
        self.test_command_input.setMaximumHeight(50)
        self.test_command_input.setPlaceholderText("Type a test command...")
        test_controls.addWidget(self.test_command_input)
        
        # Send test command button
        self.test_button = QPushButton("Send")
        self.test_button.clicked.connect(self.send_test_command)
        test_controls.addWidget(self.test_button)
        
        right_layout.addLayout(audio_controls)
        right_layout.addLayout(test_controls)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 3)  # 60% width
        main_layout.addWidget(right_panel, 2)  # 40% width
        
        # Set layout margins
        main_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setContentsMargins(0, 0, 10, 0)
        right_layout.setContentsMargins(10, 0, 0, 0)
        
        # Set default status message
        self.statusBar().showMessage("Desktop Vision Assistant Ready")
        
        # Initialize flags and state
        self.camera_running = False
        self.audio_running = False
        self.current_detections = []
        
    def start_components(self):
        """Start all system components"""
        logger.info("Starting system components")
        
        # Start camera with default camera
        self.start_camera()
        
        # Start audio processing
        self.start_audio()
        
        # Start NLP processing
        self.nlp_processor.start()
        
    def start_camera(self):
        """Start the camera and vision processing"""
        if not self.camera_running:
            logger.info("Starting camera")
            
            # Get selected camera index
            camera_index = self.camera_combo.currentIndex()
            
            # Start camera
            if self.vision_processor.start_camera(camera_index):
                # Start vision processing
                if self.vision_processor.start_processing():
                    # Start video thread
                    self.video_thread.start()
                    
                    # Update UI
                    self.camera_button.setText("Stop Camera")
                    self.camera_running = True
                    self.statusBar().showMessage("Camera started")
                else:
                    self.statusBar().showMessage("Failed to start vision processing")
            else:
                self.statusBar().showMessage("Failed to start camera")
                
    def stop_camera(self):
        """Stop the camera and vision processing"""
        if self.camera_running:
            logger.info("Stopping camera")
            
            # Stop video thread
            self.video_thread.stop()
            
            # Stop vision processing
            self.vision_processor.stop_processing()
            
            # Stop camera
            self.vision_processor.stop_camera()
            
            # Update UI
            self.camera_button.setText("Start Camera")
            self.camera_running = False
            self.statusBar().showMessage("Camera stopped")
            
            # Clear video display
            blank_image = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            self.update_frame(blank_image)
            
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()
            
    def change_camera(self):
        """Change the camera source"""
        if self.camera_running:
            # Restart camera with new source
            self.stop_camera()
            self.start_camera()
            
    def start_audio(self):
        """Start the audio processing"""
        if not self.audio_running:
            logger.info("Starting audio processing")
            
            # Start speech recognition
            self.speech_recognizer.start()
            
            # Update UI
            self.audio_button.setText("Stop Listening")
            self.audio_running = True
            self.statusBar().showMessage("Listening for commands")
            
    def stop_audio(self):
        """Stop the audio processing"""
        if self.audio_running:
            logger.info("Stopping audio processing")
            
            # Stop speech recognition
            self.speech_recognizer.stop()
            
            # Update UI
            self.audio_button.setText("Start Listening")
            self.audio_running = False
            self.statusBar().showMessage("Listening stopped")
            
    def toggle_audio(self):
        """Toggle audio on/off"""
        if self.audio_running:
            self.stop_audio()
        else:
            self.start_audio()
            
    def check_transcriptions(self):
        """Check for new transcriptions and process them"""
        transcription = self.speech_recognizer.get_transcription()
        if transcription:
            # Display transcription
            self.transcription_text.append(transcription)
            self.transcription_text.ensureCursorVisible()
            
            # Process command
            self.nlp_processor.process_command(transcription)
            
    def check_responses(self):
        """Check for new NLP responses"""
        response = self.nlp_processor.get_response()
        if response:
            action = response.get("action")
            params = response.get("params", {})
            
            # Format response based on current detections
            formatted_response = self.nlp_processor.format_detection_response(
                self.current_detections, action, params
            )
            
            # Display response
            self.response_text.append(formatted_response)
            self.response_text.ensureCursorVisible()
            
    def send_test_command(self):
        """Send a test command from the input box"""
        command = self.test_command_input.toPlainText().strip()
        if command:
            # Display command
            self.transcription_text.append(f"[Test] {command}")
            self.transcription_text.ensureCursorVisible()
            
            # Process command
            self.nlp_processor.process_command(command)
            
            # Clear input
            self.test_command_input.clear()
            
    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        """Update the video display with the latest frame"""
        if frame is None:
            return
            
        # If detection visualization is enabled, draw detections on the frame
        if self.detection_checkbox.isChecked() and self.current_detections:
            frame = self.vision_processor.draw_detections(frame, self.current_detections)
            
        # Convert frame to QImage
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        
        # Scale image to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Update label
        self.video_label.setPixmap(scaled_pixmap)
        
    @pyqtSlot(list)
    def update_detections(self, detections):
        """Update the current detections and display them"""
        self.current_detections = detections
        
        # Update objects text
        self.objects_text.clear()
        if detections:
            # Group by class
            object_counts = {}
            for det in detections:
                class_name = det['class_name']
                confidence = det['confidence']
                
                if class_name in object_counts:
                    object_counts[class_name].append(confidence)
                else:
                    object_counts[class_name] = [confidence]
                    
            # Display grouped objects
            for class_name, confidences in object_counts.items():
                avg_confidence = sum(confidences) / len(confidences)
                count = len(confidences)
                self.objects_text.append(f"{class_name}: {count} detected (avg conf: {avg_confidence:.2f})")
                
    def closeEvent(self, event):
        """Handle window close event"""
        logger.info("Closing application")
        
        # Stop components
        self.stop_camera()
        self.stop_audio()
        
        # Stop NLP processor
        self.nlp_processor.stop()
        
        # Accept close event
        event.accept()


def run_gui(speech_recognizer, vision_processor, nlp_processor):
    """Run the GUI application"""
    app = QApplication(sys.argv)
    window = DesktopVisionGUI(speech_recognizer, vision_processor, nlp_processor)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # This module shouldn't be run directly
    logger.error("This module should not be run directly")