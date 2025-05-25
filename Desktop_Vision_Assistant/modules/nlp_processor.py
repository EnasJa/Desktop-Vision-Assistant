import logging
import threading
import time
from queue import Queue
from transformers import pipeline

from config import NLP_MODEL

logger = logging.getLogger(__name__)

class NLPProcessor:
    def __init__(self, model_name=NLP_MODEL):
        logger.info(f"Initializing NLP processor with model: {model_name}")
        self.model_name = model_name
        
        # For text generation (simplified to avoid dependencies)
        try:
            self.generator = pipeline('text-generation', model=model_name)
            logger.info("Successfully loaded text generation model")
        except Exception as e:
            logger.warning(f"Could not load text generation model: {e}")
            logger.info("Using simplified text processing instead")
            self.generator = None
        
        # For command processing
        self.processing = False
        self.command_queue = Queue()
        self.response_queue = Queue()
        self.processing_thread = None
        
        # Define command templates and actions
        self.commands = {
            "identify": {"keywords": ["identify", "what", "objects", "see"], "action": "identify_objects"},
            "count": {"keywords": ["count", "how many"], "action": "count_objects"},
            "describe": {"keywords": ["describe", "tell", "about", "scene"], "action": "describe_scene"},
            "find": {"keywords": ["find", "locate", "where", "is"], "action": "find_object"},
            "track": {"keywords": ["track", "follow"], "action": "track_object"}
        }
        
    def start(self):
        """Start the NLP processing thread"""
        if self.processing:
            logger.warning("NLP processing already running")
            return
            
        logger.info("Starting NLP processing")
        self.processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop(self):
        """Stop the NLP processing thread"""
        logger.info("Stopping NLP processing")
        self.processing = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
            
    def process_command(self, text):
        """Add a command to the processing queue"""
        if text and text.strip():
            logger.info(f"Adding command to queue: {text}")
            self.command_queue.put(text)
            
    def get_response(self):
        """Get the latest response if available"""
        if not self.response_queue.empty():
            return self.response_queue.get()
        return None
        
    def _processing_loop(self):
        """Main processing loop for commands"""
        logger.info("NLP processing loop started")
        while self.processing:
            if not self.command_queue.empty():
                command_text = self.command_queue.get()
                response = self._parse_command(command_text)
                self.response_queue.put(response)
            time.sleep(0.1)
            
    def _parse_command(self, text):
        """Parse command text and determine action"""
        logger.info(f"Parsing command: {text}")
        text = text.lower()
        
        # Check for command matches
        best_match = None
        best_score = 0
        
        for cmd_name, cmd_data in self.commands.items():
            score = sum([1 for keyword in cmd_data["keywords"] if keyword in text])
            if score > best_score:
                best_score = score
                best_match = cmd_data["action"]
                
        if best_match and best_score > 0:
            logger.info(f"Command matched: {best_match} (score: {best_score})")
            return {
                "action": best_match,
                "original_text": text,
                "params": self._extract_params(text, best_match)
            }
        else:
            logger.info("No command matched, treating as general query")
            return {
                "action": "general_query",
                "original_text": text,
                "params": {"query": text}
            }
            
    def _extract_params(self, text, action):
        """Extract parameters for specific actions"""
        params = {}
        
        if action == "identify_objects":
            # No specific parameters needed
            pass
            
        elif action == "count_objects":
            # Try to extract what to count
            object_to_count = None
            words = text.split()
            for i, word in enumerate(words):
                if word in ["count", "many"]:
                    # Look for object after these words
                    if i + 1 < len(words):
                        object_to_count = words[i + 1]
                        break
                        
            params["object_type"] = object_to_count
            
        elif action == "find_object" or action == "track_object":
            # Extract the object to find/track
            object_to_find = None
            
            # List of words to skip in extraction
            skip_words = ["find", "locate", "where", "is", "the", "a", "an", "track", "follow"]
            
            # Simple extraction (can be improved)
            words = text.split()
            for i, word in enumerate(words):
                if word in skip_words:
                    continue
                object_to_find = word
                # Check if next word should be included
                if i + 1 < len(words) and words[i + 1] not in skip_words:
                    object_to_find += " " + words[i + 1]
                break
                
            params["object_type"] = object_to_find
            
        elif action == "describe_scene":
            # No specific parameters needed
            pass
            
        return params
        
    def generate_text(self, prompt, max_length=100):
        """Generate text based on a prompt"""
        try:
            if self.generator:
                logger.info(f"Generating text for prompt: {prompt}")
                generated = self.generator(prompt, max_length=max_length, num_return_sequences=1)
                return generated[0]['generated_text']
            else:
                # Simple text generation fallback
                logger.info("Using simplified text generation")
                return f"{prompt} (simplified response)"
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return prompt
            
    def format_detection_response(self, detections, action, params=None):
        """Format the detection results into a natural language response"""
        if not detections:
            return "I don't see any objects in the current view."
            
        if action == "identify_objects":
            # List all detected objects
            object_counts = {}
            for det in detections:
                class_name = det['class_name']
                if class_name in object_counts:
                    object_counts[class_name] += 1
                else:
                    object_counts[class_name] = 1
                    
            if not object_counts:
                return "I don't see any recognizable objects."
                
            response = "I can see: "
            items = []
            for obj, count in object_counts.items():
                if count > 1:
                    items.append(f"{count} {obj}s")
                else:
                    items.append(f"a {obj}")
                    
            response += ", ".join(items)
            return response
            
        elif action == "count_objects":
            # Count specific objects or all objects
            object_type = params.get("object_type") if params else None
            
            if object_type:
                # Count specific object type
                count = sum(1 for det in detections if object_type in det['class_name'].lower())
                if count == 0:
                    return f"I don't see any {object_type} in the current view."
                elif count == 1:
                    return f"I see 1 {object_type} in the current view."
                else:
                    return f"I see {count} {object_type}s in the current view."
            else:
                # Count all objects
                return f"I see {len(detections)} objects in the current view."
                
        elif action == "find_object":
            # Locate a specific object
            object_type = params.get("object_type") if params else None
            
            if not object_type:
                return "Please specify what object you're looking for."
                
            # Find objects matching the type
            matching_objects = [det for det in detections if object_type in det['class_name'].lower()]
            
            if not matching_objects:
                return f"I don't see any {object_type} in the current view."
                
            # Get the first matching object's position
            obj = matching_objects[0]
            bbox = obj['bbox']
            x_center = (bbox[0] + bbox[2]) // 2
            y_center = (bbox[1] + bbox[3]) // 2
            
            # Determine position in frame
            position = ""
            if y_center < 240:  # Top third
                position += "top "
            elif y_center > 480:  # Bottom third
                position += "bottom "
                
            if x_center < 213:  # Left third
                position += "left"
            elif x_center > 426:  # Right third
                position += "right"
            else:
                position += "center"
                
            return f"I found a {obj['class_name']} in the {position} of the view."
            
        elif action == "describe_scene":
            # Generate a description of the scene
            if len(detections) == 0:
                return "I don't see any objects to describe."
                
            # Count objects by type
            object_counts = {}
            for det in detections:
                class_name = det['class_name']
                if class_name in object_counts:
                    object_counts[class_name] += 1
                else:
                    object_counts[class_name] = 1
                    
            # Create scene description
            description = "I can see "
            items = []
            
            for obj, count in object_counts.items():
                if count > 1:
                    items.append(f"{count} {obj}s")
                else:
                    items.append(f"a {obj}")
                    
            description += ", ".join(items)
            description += " in the current scene."
            
            return description
            
        elif action == "track_object":
            # Provide info for tracking a specific object
            object_type = params.get("object_type") if params else None
            
            if not object_type:
                return "Please specify what object you want me to track."
                
            # Find objects matching the type
            matching_objects = [det for det in detections if object_type in det['class_name'].lower()]
            
            if not matching_objects:
                return f"I don't see any {object_type} to track in the current view."
                
            # Get the first matching object's position
            obj = matching_objects[0]
            bbox = obj['bbox']
            x_center = (bbox[0] + bbox[2]) // 2
            y_center = (bbox[1] + bbox[3]) // 2
            
            # Determine position in frame
            position = ""
            if y_center < 240:  # Top third
                position += "top "
            elif y_center > 480:  # Bottom third
                position += "bottom "
                
            if x_center < 213:  # Left third
                position += "left"
            elif x_center > 426:  # Right third
                position += "right"
            else:
                position += "center"
                
            return f"Now tracking the {obj['class_name']} in the {position} of the view."
            
        else:  # General query or unknown action
            return "I'm not sure how to respond to that command with the current view."


# Simple test function
def test_nlp_processor():
    nlp = NLPProcessor()
    
    # Test command parsing
    test_commands = [
        "What objects do you see?",
        "Count how many people are there",
        "Describe the scene",
        "Find the book",
        "Track the car"
    ]
    
    for cmd in test_commands:
        result = nlp._parse_command(cmd)
        print(f"Command: '{cmd}'")
        print(f"  Action: {result['action']}")
        print(f"  Params: {result['params']}")
        
    # Test response formatting
    test_detections = [
        {'class_id': 0, 'class_name': 'person', 'confidence': 0.9, 'bbox': [100, 150, 200, 300]},
        {'class_id': 0, 'class_name': 'person', 'confidence': 0.85, 'bbox': [300, 200, 400, 350]},
        {'class_id': 2, 'class_name': 'car', 'confidence': 0.7, 'bbox': [50, 50, 150, 100]},
        {'class_id': 62, 'class_name': 'chair', 'confidence': 0.65, 'bbox': [400, 300, 450, 400]}
    ]
    
    for cmd in test_commands:
        result = nlp._parse_command(cmd)
        response = nlp.format_detection_response(
            test_detections, 
            result['action'], 
            result['params']
        )
        print(f"Command: '{cmd}'")
        print(f"  Response: {response}")
        print()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Test the NLP processor
    test_nlp_processor()
