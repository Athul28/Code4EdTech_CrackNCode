from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import easyocr
from PIL import Image
import io
import re
from typing import List, Dict, Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="OMR Sheet Processor", version="1.0.0")

# Global variables for model and OCR reader
model = None
ocr_reader = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model and OCR reader on startup"""
    global model, ocr_reader
    
    try:
        # Load the trained YOLOv8 model
        model_path = "./models/best.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model = YOLO(model_path)
        logger.info("YOLO model loaded successfully")
        
        # Initialize EasyOCR reader for English text
        ocr_reader = easyocr.Reader(['en'])
        logger.info("OCR reader initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded image bytes to OpenCV format"""
    try:
        # Convert bytes to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_bgr
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image format")

def parse_answer_key(csv_content: str) -> Dict[int, str]:
    """Parse CSV answer key into dictionary"""
    try:
        # Create DataFrame from CSV string
        csv_data = io.StringIO(csv_content)
        df = pd.read_csv(csv_data)
        
        # Ensure required columns exist
        if 'question_number' not in df.columns or 'answer' not in df.columns:
            raise ValueError("CSV must contain 'question_number' and 'answer' columns")
        
        # Convert to dictionary
        answer_key = {}
        for _, row in df.iterrows():
            q_num = int(row['question_number'])
            answer = str(row['answer']).upper().strip()
            answer_key[q_num] = answer
        
        logger.info(f"Parsed answer key with {len(answer_key)} questions")
        return answer_key
    
    except Exception as e:
        logger.error(f"Error parsing answer key: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid CSV format or content")

def detect_bubbles(image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
    """Use YOLO model to detect answer bubbles"""
    try:
        # Run prediction
        results = model(image, conf=confidence_threshold)
        
        detections = []
        boxes = results[0].boxes
        
        if boxes is not None:
            # Get box coordinates, confidences, and class IDs
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, class_id) in enumerate(zip(xyxy, confidences, class_ids)):
                x1, y1, x2, y2 = box.astype(int)
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class_id': int(class_id),
                    'class_name': model.names.get(class_id, f"Class_{class_id}")
                }
                detections.append(detection)
        
        logger.info(f"Detected {len(detections)} bubbles")
        return detections
    
    except Exception as e:
        logger.error(f"Error detecting bubbles: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in bubble detection")

def extract_question_number(image: np.ndarray, bbox: List[int]) -> Optional[int]:
    """Extract question number using OCR from a region near the bounding box"""
    try:
        x1, y1, x2, y2 = bbox
        
        # Expand the region to include potential question number area
        # Typically question numbers are to the left of answer bubbles
        expanded_x1 = max(0, x1 - 100)  # Look 100 pixels to the left
        expanded_y1 = max(0, y1 - 20)   # Look 20 pixels above
        expanded_x2 = x1 + 50           # Include some area to the right
        expanded_y2 = y2 + 20           # Look 20 pixels below
        
        # Crop the region
        question_region = image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
        
        # Use OCR to extract text
        ocr_results = ocr_reader.readtext(question_region)
        
        # Look for question numbers in the OCR results
        for (bbox_ocr, text, confidence) in ocr_results:
            if confidence > 0.5:  # Only consider high-confidence results
                # Extract numbers from the text
                numbers = re.findall(r'\b\d+\b', text)
                if numbers:
                    # Return the first number found (likely the question number)
                    return int(numbers[0])
        
        return None
    
    except Exception as e:
        logger.error(f"Error extracting question number: {str(e)}")
        return None

def determine_selected_option(image: np.ndarray, detections: List[Dict]) -> Dict[int, str]:
    """Determine which option is selected for each question"""
    try:
        # Group detections by question number
        question_bubbles = {}
        
        for detection in detections:
            # Extract question number for this detection
            q_num = extract_question_number(image, detection['bbox'])
            
            if q_num is not None:
                if q_num not in question_bubbles:
                    question_bubbles[q_num] = []
                question_bubbles[q_num].append(detection)
        
        # For each question, determine the selected option
        selected_answers = {}
        
        for q_num, bubbles in question_bubbles.items():
            if not bubbles:
                continue
            
            # Sort bubbles by x-coordinate (left to right = A, B, C, D)
            bubbles.sort(key=lambda x: x['bbox'][0])
            
            # Analyze each bubble to see if it's filled
            filled_options = []
            
            for i, bubble in enumerate(bubbles):
                x1, y1, x2, y2 = bubble['bbox']
                bubble_region = image[y1:y2, x1:x2]
                
                # Convert to grayscale
                if len(bubble_region.shape) == 3:
                    gray_bubble = cv2.cvtColor(bubble_region, cv2.COLOR_BGR2GRAY)
                else:
                    gray_bubble = bubble_region
                
                # Calculate the mean intensity (lower means darker/filled)
                mean_intensity = np.mean(gray_bubble)
                
                # Threshold to determine if bubble is filled
                # This threshold may need adjustment based on your specific images
                if mean_intensity < 200:  # Adjust this value as needed
                    option_letter = chr(ord('A') + i)
                    filled_options.append(option_letter)
            
            # Handle multiple selections or no selection
            if len(filled_options) == 1:
                selected_answers[q_num] = filled_options[0]
            elif len(filled_options) > 1:
                selected_answers[q_num] = "MULTIPLE"
            else:
                selected_answers[q_num] = "NONE"
        
        return selected_answers
    
    except Exception as e:
        logger.error(f"Error determining selected options: {str(e)}")
        return {}

def save_bounding_boxes_visualization(image: np.ndarray, detections: List[Dict], filename: str = "debug_bounding_boxes.jpg") -> str:
    """Save image with bounding boxes drawn for debugging"""
    try:
        # Create a copy of the image for drawing
        vis_image = image.copy()
        
        # Create output directory if it doesn't exist
        output_dir = "./debug_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Draw bounding boxes
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with detection info
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(vis_image, 
                        (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), 
                        (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(vis_image, label, 
                      (x1, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw detection number
            cv2.putText(vis_image, str(i+1), 
                      (x1-15, y1+15), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Save the visualization
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, vis_image)
        
        logger.info(f"🐛 Bounding boxes visualization saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving bounding boxes visualization: {str(e)}")
        return None

def save_debug_visualization_with_ocr(image: np.ndarray, detections: List[Dict], selected_answers: Dict[int, str], filename: str = "debug_omr_with_ocr.jpg") -> str:
    """Save image with bounding boxes, OCR detected text, and answer selections for debugging"""
    try:
        # Create a copy of the image for drawing
        vis_image = image.copy()
        
        # Create output directory if it doesn't exist
        output_dir = "./debug_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Track question numbers and their positions for OCR text placement
        question_positions = {}
        
        # Draw bounding boxes and collect question information
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Extract question number for this detection
            q_num = extract_question_number(image, detection['bbox'])
            
            # Choose color based on detection type
            color = (0, 255, 0)  # Green for regular detections
            if q_num and q_num in selected_answers:
                if selected_answers[q_num] == "MULTIPLE":
                    color = (0, 165, 255)  # Orange for multiple selections
                elif selected_answers[q_num] == "NONE":
                    color = (0, 0, 255)  # Red for no selection
                else:
                    color = (255, 0, 0)  # Blue for single selection
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with detection info
            label = f"{class_name}: {confidence:.2f}"
            if q_num:
                label += f" Q{q_num}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            
            # Draw label background
            cv2.rectangle(vis_image, 
                        (x1, y1 - label_size[1] - 8), 
                        (x1 + label_size[0], y1), 
                        color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, 
                      (x1, y1 - 4), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Store question position for OCR text
            if q_num and q_num not in question_positions:
                question_positions[q_num] = (x1 - 120, y1 + 15)
        
        # Draw OCR detected text and answer information
        for q_num, (text_x, text_y) in question_positions.items():
            # Get the detected answer for this question
            detected_answer = selected_answers.get(q_num, "NONE")
            
            # Create OCR text to display
            ocr_text = f"Q{q_num}: {detected_answer}"
            
            # Choose text color based on answer status
            text_color = (0, 255, 0)  # Green for normal answers
            if detected_answer == "MULTIPLE":
                text_color = (0, 165, 255)  # Orange
            elif detected_answer == "NONE":
                text_color = (0, 0, 255)  # Red
            
            # Draw text background for better visibility
            text_size = cv2.getTextSize(ocr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_image, 
                        (text_x - 5, text_y - text_size[1] - 5), 
                        (text_x + text_size[0] + 5, text_y + 5), 
                        (0, 0, 0), -1)
            
            # Draw OCR text
            cv2.putText(vis_image, ocr_text, 
                      (text_x, text_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Add legend
        legend_y = 30
        legend_items = [
            ("Legend:", (255, 255, 255)),
            ("Blue: Single Selection", (255, 0, 0)),
            ("Green: Regular Detection", (0, 255, 0)),
            ("Orange: Multiple Selection", (0, 165, 255)),
            ("Red: No Selection", (0, 0, 255))
        ]
        
        for i, (text, color) in enumerate(legend_items):
            y_pos = legend_y + (i * 25)
            # Background for legend text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis_image, (10, y_pos - text_size[1] - 5), 
                        (15 + text_size[0], y_pos + 5), (0, 0, 0), -1)
            cv2.putText(vis_image, text, (15, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save the visualization
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, vis_image)
        
        logger.info(f"🔍 Debug visualization with OCR saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving debug visualization with OCR: {str(e)}")
        return None

def evaluate_answers(selected_answers: Dict[int, str], answer_key: Dict[int, str]) -> Dict:
    """Compare selected answers with answer key and return results"""
    try:
        results = {
            'total_questions': len(answer_key),
            'attempted': 0,
            'correct': 0,
            'incorrect': 0,
            'not_attempted': 0,
            'multiple_selections': 0,
            'score_percentage': 0.0,
            'detailed_results': []
        }
        
        for q_num, correct_answer in answer_key.items():
            selected = selected_answers.get(q_num, "NONE")
            
            result_detail = {
                'question_number': q_num,
                'correct_answer': correct_answer,
                'selected_answer': selected,
                'status': 'not_attempted'
            }
            
            if selected == "NONE":
                results['not_attempted'] += 1
                result_detail['status'] = 'not_attempted'
            elif selected == "MULTIPLE":
                results['multiple_selections'] += 1
                results['attempted'] += 1
                result_detail['status'] = 'multiple_selection'
            else:
                results['attempted'] += 1
                if selected == correct_answer:
                    results['correct'] += 1
                    result_detail['status'] = 'correct'
                else:
                    results['incorrect'] += 1
                    result_detail['status'] = 'incorrect'
            
            results['detailed_results'].append(result_detail)
        
        # Calculate score percentage
        if results['total_questions'] > 0:
            results['score_percentage'] = (results['correct'] / results['total_questions']) * 100
        
        return results
    
    except Exception as e:
        logger.error(f"Error evaluating answers: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in answer evaluation")

@app.post("/process_omr")
async def process_omr_sheet(
    image: UploadFile = File(..., description="OMR sheet image"),
    answer_key: str = Form(..., description="Answer key in CSV format with columns: question_number, answer")
):
    """
    Process an OMR sheet image and compare with answer key
    
    Args:
        image: OMR sheet image file
        answer_key: CSV string with question numbers and correct answers
    
    Returns:
        JSON response with evaluation results
    """
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and preprocess image
        image_bytes = await image.read()
        processed_image = preprocess_image(image_bytes)
        
        # Parse answer key
        answer_key_dict = parse_answer_key(answer_key)
        
        # Detect bubbles using YOLO model
        detections = detect_bubbles(processed_image)
        
        # Determine selected answers
        selected_answers = determine_selected_option(processed_image, detections)
        
        # Evaluate answers
        evaluation_results = evaluate_answers(selected_answers, answer_key_dict)
        
        # Save debug visualization with bounding boxes and OCR text
        debug_filename = f"debug_omr_with_ocr_{image.filename}.jpg" if image.filename else "debug_omr_with_ocr.jpg"
        debug_path = save_debug_visualization_with_ocr(processed_image, detections, selected_answers, debug_filename)
        
        # Add debug information to results
        evaluation_results['debug_info'] = {
            'total_detections': len(detections),
            'debug_image_path': debug_path,
            'message': 'Debug visualization saved with bounding boxes and OCR detected text'
        }
        
        logger.info(f"Processing completed. Score: {evaluation_results['score_percentage']:.2f}%")
        
        return JSONResponse(content=evaluation_results)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None, "ocr_ready": ocr_reader is not None}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "OMR Sheet Processor API",
        "version": "1.0.0",
        "endpoints": {
            "/process_omr": "POST - Process OMR sheet with answer key",
            "/debug_omr": "POST - Process OMR sheet with detailed debug information",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
