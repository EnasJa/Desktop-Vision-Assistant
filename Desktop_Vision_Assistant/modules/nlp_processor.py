import logging
import threading
import time
from queue import Queue
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from config import NLP_MODEL

logger = logging.getLogger(__name__)

class NLPProcessor:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):  # Default to a model that works without sentencepiece
        logger.info(f"Initializing NLP processor with conversational model: {model_name}")
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Use models that don't require sentencepiece:
        # "microsoft/DialoGPT-medium" - Good for conversations
        # "microsoft/DialoGPT-small" - Smaller version
        # "facebook/blenderbot-400M-distill" - Alternative (but might need sentencepiece)
        # "gpt2" - Simple and reliable
        
        try:
            logger.info("Loading tokenizer and model...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Create text generation pipeline
            self.generator = pipeline(
                'text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info("Conversational model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading conversational model: {e}")
            logger.info("Falling back to simplified text processing")
            self.model = None
            self.tokenizer = None
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
            "track": {"keywords": ["track", "follow"], "action": "track_object"},
            "chat": {"keywords": ["chat", "talk", "conversation"], "action": "general_chat"}
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
        text_lower = text.lower()
        
        # Check for command matches
        best_match = None
        best_score = 0
        
        for cmd_name, cmd_data in self.commands.items():
            score = sum([1 for keyword in cmd_data["keywords"] if keyword in text_lower])
            if score > best_score:
                best_score = score
                best_match = cmd_data["action"]
                
        if best_match and best_score > 0:
            logger.info(f"Command matched: {best_match} (score: {best_score})")
            return {
                "action": best_match,
                "original_text": text,
                "params": self._extract_params(text_lower, best_match)
            }
        else:
            logger.info("No specific command matched, treating as general chat")
            return {
                "action": "general_chat",
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
            
        elif action == "general_chat":
            params["query"] = text
            
        return params
        
    def generate_llama_response(self, prompt, max_length=150):
        """Generate response using LLaMA model"""
        try:
            if self.generator is None:
                return f"I understand you said: '{prompt}'. However, the AI model is not available right now."
            
            logger.info(f"Generating LLaMA response for: {prompt}")
            
            # Format prompt for better conversation
            formatted_prompt = f"Human: {prompt}\nAssistant:"
            
            # Generate response
            outputs = self.generator(
                formatted_prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            generated_text = outputs[0]['generated_text']
            
            # Remove the prompt from the response
            response = generated_text.replace(formatted_prompt, "").strip()
            
            # Clean up the response
            if response.startswith("Assistant:"):
                response = response[10:].strip()
            
            # Limit response length and clean up
            response = response.split('\n')[0]  # Take first line only
            response = response[:200]  # Limit length
            
            return response if response else "I'm here to help you with vision-related tasks."
            
        except Exception as e:
            logger.error(f"Error generating LLaMA response: {e}")
            return f"I understand your message: '{prompt}'. How can I help you with vision analysis?"
            
    def format_detection_response(self, detections, action, params=None):
        """Format the detection results into a natural language response"""
        if not detections:
            if action == "general_chat":
                query = params.get("query", "") if params else ""
                return self.generate_llama_response(f"{query}. Currently, I don't see any objects in the camera view.")
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
            # Generate a description of the scene using LLaMA
            object_counts = {}
            for det in detections:
                class_name = det['class_name']
                if class_name in object_counts:
                    object_counts[class_name] += 1
                else:
                    object_counts[class_name] = 1
            
            objects_list = []
            for obj, count in object_counts.items():
                if count > 1:
                    objects_list.append(f"{count} {obj}s")
                else:
                    objects_list.append(f"a {obj}")
            
            scene_prompt = f"I can see {', '.join(objects_list)} in the scene. Please describe what this scene might be."
            return self.generate_llama_response(scene_prompt)
            
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
            
        elif action == "general_chat":
            # Use LLaMA for general conversation
            query = params.get("query", "") if params else ""
            
            # Include information about what's visible
            object_names = [det['class_name'] for det in detections]
            unique_objects = list(set(object_names))
            
            context_prompt = f"{query}. For context, I can currently see these objects in the camera: {', '.join(unique_objects)}."
            return self.generate_llama_response(context_prompt)
            
        else:  # Unknown action
            return "I'm not sure how to respond to that command with the current view."


# Test function for LLaMA integration
def test_llama_nlp():
    nlp = NLPProcessor()
    
    test_commands = [
        "What do you see?",
        "Tell me about the scene",
        "How are you today?",
        "What's the weather like?"
    ]
    
    for cmd in test_commands:
        result = nlp._parse_command(cmd)
        print(f"Command: '{cmd}'")
        print(f"  Action: {result['action']}")
        print(f"  Params: {result['params']}")
        
        if result['action'] == 'general_chat':
            response = nlp.generate_llama_response(cmd)
            print(f"  LLaMA Response: {response}")
        print()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Test the LLaMA NLP processor
    test_llama_nlp()