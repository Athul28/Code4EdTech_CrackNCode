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
import requests
import os
from config.cloudinary import cloudinary
from config.gemini import genai
from dotenv import load_dotenv

load_dotenv()

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
            answer = row['answer']
            if isinstance(answer, list):
                answers = [str(a).upper().strip() for a in answer]
            else:
                answers = [str(answer).upper().strip()]
            answer_key[q_num] = answers
        
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

def save_bounding_boxes_visualization(image: np.ndarray, detections: List[Dict], ocr_results: Dict = None, filename: str = "debug_bounding_boxes.jpg") -> str:
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
            
            # Draw OCR text or detection number
            ocr_text = "?"
            if ocr_results and i < len(ocr_results):
                extracted_q_num = ocr_results[i].get('extracted_question_number')
                if extracted_q_num is not None:
                    ocr_text = str(extracted_q_num)
                else:
                    # Try to get the best OCR text from raw results
                    raw_results = ocr_results[i].get('ocr_raw_results', [])
                    if raw_results:
                        # Get the text with highest confidence
                        best_result = max(raw_results, key=lambda x: x['confidence'])
                        if best_result['confidence'] > 0.3:  # Lower threshold for display
                            ocr_text = best_result['text'][:10]  # Limit length
            
            cv2.putText(vis_image, ocr_text, 
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

def evaluate_answers(selected_answers: Dict[int, str], answer_key: Dict[int, List[str]]) -> Dict:
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
                if selected in correct_answer:
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

@app.post("/upload/answers/{examId}")
async def upload_answer_key(
    file: UploadFile = File(..., description="Answer key CSV file")
):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="File must be a CSV")

        # Read file content
        csv_bytes = await file.read()
        csv_text = csv_bytes.decode("utf-8")

        # Step 1: Process with Gemini
        prompt = f"""
        You are given an exam answer key in CSV format.
        Convert it into a CSV with exactly 2 columns:
        - question_number (integer)
        - answer (array of correct answers, like ['A'] or ['A','C'])
        
        Input CSV:
        {csv_text}
        """

        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_model.generate_content(prompt)

        processed_csv = response.text.strip()

        # Step 2: Save processed CSV locally before upload
        temp_path = f"./{examId}.csv"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(processed_csv)

        # Step 3: Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            temp_path,
            resource_type="raw",
            folder="exam",
            public_id=examId,
            overwrite=True
        )

        # Get secure URL
        csv_url = upload_result["secure_url"]

        return {"message": "Answer key uploaded successfully", "examId": examId, "url": csv_url}

    except Exception as e:
        logger.error(f"Error uploading answer key: {str(e)}")
        raise HTTPException(status_code=500, detail="Error uploading answer key")


@app.post("/process_omr")
async def process_omr_sheet(
    examId: str = Form(..., description="Exam ID for fetching answer key"),
    image: UploadFile = File(..., description="OMR sheet image")
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
        
        csv_url = f"https://res.cloudinary.com/{os.getenv('CLOUDINARY_CLOUD')}/raw/upload/exam/{examId}.csv"
        response = requests.get(csv_url)

        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Answer key not found for this examId")

        answer_key_text = response.text
        answer_key_dict = parse_answer_key(answer_key_text)
        
        # Detect bubbles using YOLO model
        detections = detect_bubbles(processed_image)
        
        # Determine selected answers
        selected_answers = determine_selected_option(processed_image, detections)
        
        # Evaluate answers
        evaluation_results = evaluate_answers(selected_answers, answer_key_dict)
        
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

@app.post("/debug_omr")
async def debug_omr_sheet(
    image: UploadFile = File(..., description="OMR sheet image"),
    examId: str = Form(..., description="Unique exam identifier"),
):
    """
    DEBUG VERSION: Process an OMR sheet with detailed logging and debug information
    
    This endpoint provides comprehensive debugging information for each step:
    - Image preprocessing details
    - YOLO model predictions with all detections
    - OCR results for each detected region
    - Bubble analysis details
    - Step-by-step answer determination
    
    Args:
        image: OMR sheet image file
        answer_key: CSV string with question numbers and correct answers
    
    Returns:
        JSON response with evaluation results AND detailed debug information
    """
    debug_info = {
        "step_1_image_preprocessing": {},
        "step_2_model_predictions": {},
        "step_3_ocr_detections": {},
        "step_4_bubble_analysis": {},
        "step_5_answer_evaluation": {},
        "final_results": {}
    }
    
    try:
        logger.info("🐛 DEBUG MODE: Starting OMR processing with detailed logging")
        
        # STEP 1: Image Preprocessing
        logger.info("🐛 STEP 1: Image preprocessing")
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await image.read()
        processed_image = preprocess_image(image_bytes)
        
        debug_info["step_1_image_preprocessing"] = {
            "original_filename": image.filename,
            "file_size_bytes": len(image_bytes),
            "content_type": image.content_type,
            "processed_image_shape": processed_image.shape,
            "processed_image_dtype": str(processed_image.dtype)
        }
        logger.info(f"🐛 Image processed: {processed_image.shape} shape, {processed_image.dtype} dtype")
        
        # STEP 2: Parse Answer Key
        logger.info("🐛 STEP 2: Parsing answer key")
        csv_url = f"https://res.cloudinary.com/{os.getenv('CLOUDINARY_CLOUD')}/raw/upload/exam/{examId}.csv"
        response = requests.get(csv_url)

        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Answer key not found for this examId")
        answer_key = response.text
        answer_key_dict = parse_answer_key(answer_key)
        debug_info["step_2_answer_key"] = {
            "total_questions_in_key": len(answer_key_dict),
            "answer_key_sample": dict(list(answer_key_dict.items())[:5]),  # First 5 items
            "all_answers": answer_key_dict
        }
        logger.info(f"🐛 Answer key parsed: {len(answer_key_dict)} questions")
        
        # STEP 3: YOLO Model Predictions
        logger.info("🐛 STEP 3: Running YOLO model predictions")
        results = model(processed_image, conf=0.5)
        
        # Detailed model prediction logging
        model_debug = {
            "confidence_threshold": 0.5,
            "total_results": len(results),
            "detections": []
        }
        
        detections = []
        boxes = results[0].boxes
        
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            
            logger.info(f"🐛 Raw YOLO output: {len(xyxy)} detections")
            
            for i, (box, conf, class_id) in enumerate(zip(xyxy, confidences, class_ids)):
                x1, y1, x2, y2 = box.astype(int)
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class_id': int(class_id),
                    'class_name': model.names.get(class_id, f"Class_{class_id}"),
                    'bbox_width': x2 - x1,
                    'bbox_height': y2 - y1,
                    'bbox_area': (x2 - x1) * (y2 - y1)
                }
                detections.append(detection)
                model_debug["detections"].append(detection)
                
                # logger.info(f"🐛 Detection {i+1}: {detection['class_name']} at [{x1},{y1},{x2},{y2}] conf={conf:.3f}")
        
        debug_info["step_2_model_predictions"] = model_debug
        logger.info(f"🐛 Total detections after filtering: {len(detections)}")
        
        # STEP 4: OCR Processing for Each Detection
        logger.info("🐛 STEP 4: OCR processing for question numbers")
        ocr_debug = {
            "total_detections_processed": len(detections),
            "ocr_results": [],
            "question_number_mappings": {}
        }
        
        question_bubbles = {}
        
        for idx, detection in enumerate(detections):
            logger.info(f"🐛 Processing detection {idx+1}/{len(detections)}")
            
            # Extract question number with detailed logging
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Expand the region for OCR
            expanded_x1 = max(0, x1 - 100)
            expanded_y1 = max(0, y1 - 20)
            expanded_x2 = x1 + 50
            expanded_y2 = y2 + 20
            
            question_region = processed_image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
            
            ocr_detection_debug = {
                "detection_index": idx,
                "original_bbox": bbox,
                "expanded_bbox": [expanded_x1, expanded_y1, expanded_x2, expanded_y2],
                "region_shape": question_region.shape,
                "ocr_raw_results": [],
                "extracted_question_number": None
            }
            
            try:
                # Run OCR
                ocr_results = ocr_reader.readtext(question_region)
                
                # logger.info(f"🐛 OCR found {len(ocr_results)} text regions")
                
                for ocr_result in ocr_results:
                    bbox_ocr, text, confidence = ocr_result
                    ocr_detection_debug["ocr_raw_results"].append({
                        "text": text,
                        "confidence": float(confidence),
                        "bbox": bbox_ocr
                    })
                    # logger.info(f"🐛 OCR text: '{text}' (conf: {confidence:.3f})")
                    
                    if confidence > 0.5:
                        numbers = re.findall(r'\b\d+\b', text)
                        if numbers:
                            q_num = int(numbers[0])
                            ocr_detection_debug["extracted_question_number"] = q_num
                            
                            if q_num not in question_bubbles:
                                question_bubbles[q_num] = []
                            question_bubbles[q_num].append(detection)
                            
                            logger.info(f"🐛 ✅ Question number {q_num} extracted successfully")
                            break
                
                if ocr_detection_debug["extracted_question_number"] is None:
                    logger.info(f"🐛 ❌ No question number found for detection {idx+1}")
                    
            except Exception as e:
                logger.error(f"🐛 OCR error for detection {idx+1}: {str(e)}")
                ocr_detection_debug["error"] = str(e)
            
            ocr_debug["ocr_results"].append(ocr_detection_debug)
        
        # Map question numbers to detections
        for q_num, bubbles in question_bubbles.items():
            ocr_debug["question_number_mappings"][q_num] = len(bubbles)
        
        debug_info["step_3_ocr_detections"] = ocr_debug
        logger.info(f"🐛 Question mapping complete: {len(question_bubbles)} questions found")
        
        # Save bounding boxes visualization with OCR results
        bbox_vis_path = save_bounding_boxes_visualization(processed_image, detections, 
                                                        ocr_debug["ocr_results"],
                                                        f"debug_bounding_boxes_{image.filename}.jpg")
        debug_info["step_2_model_predictions"]["visualization_saved_to"] = bbox_vis_path
        
        # Display detected question numbers and their options
        logger.info("🐛 📋 DETECTED QUESTION NUMBERS AND OPTIONS:")
        logger.info("🐛 " + "="*60)
        
        question_summary = {}
        for q_num, bubbles in sorted(question_bubbles.items()):
            # Sort bubbles by x-coordinate to determine option order
            sorted_bubbles = sorted(bubbles, key=lambda x: x['bbox'][0])
            options = []
            
            for i, bubble in enumerate(sorted_bubbles):
                option_letter = chr(ord('A') + i)
                bbox = bubble['bbox']
                confidence = bubble['confidence']
                options.append({
                    "option": option_letter,
                    "bbox": bbox,
                    "confidence": confidence
                })
            
            question_summary[q_num] = options
            
            # Log the question and its options
            options_str = ", ".join([f"{opt['option']}[{opt['bbox']}]" for opt in options])
            logger.info(f"🐛 Question {q_num:2d}: {len(options)} options detected -> {options_str}")
        
        # Add question summary to debug info
        debug_info["step_3_ocr_detections"]["question_options_summary"] = question_summary
        logger.info("🐛 " + "="*60)
        
        # STEP 5: Bubble Analysis and Answer Determination
        logger.info("🐛 STEP 5: Bubble analysis and answer determination")
        bubble_debug = {
            "questions_analyzed": {},
            "intensity_threshold": 200,
            "selected_answers": {}
        }
        
        selected_answers = {}
        
        for q_num, bubbles in question_bubbles.items():
            logger.info(f"🐛 Analyzing question {q_num} with {len(bubbles)} bubbles")
            
            question_analysis = {
                "total_bubbles": len(bubbles),
                "bubble_details": [],
                "sorted_by_position": True,
                "filled_options": [],
                "final_answer": "NONE"
            }
            
            if not bubbles:
                continue
            
            # Sort bubbles by x-coordinate
            bubbles.sort(key=lambda x: x['bbox'][0])
            
            filled_options = []
            
            for i, bubble in enumerate(bubbles):
                x1, y1, x2, y2 = bubble['bbox']
                bubble_region = processed_image[y1:y2, x1:x2]
                
                # Convert to grayscale
                if len(bubble_region.shape) == 3:
                    gray_bubble = cv2.cvtColor(bubble_region, cv2.COLOR_BGR2GRAY)
                else:
                    gray_bubble = bubble_region
                
                mean_intensity = np.mean(gray_bubble)
                is_filled = mean_intensity < 200
                option_letter = chr(ord('A') + i)
                
                bubble_detail = {
                    "option": option_letter,
                    "position_index": i,
                    "bbox": [x1, y1, x2, y2],
                    "mean_intensity": float(mean_intensity),
                    "is_filled": is_filled,
                    "region_shape": bubble_region.shape
                }
                
                question_analysis["bubble_details"].append(bubble_detail)
                
                logger.info(f"🐛 Q{q_num} Option {option_letter}: intensity={mean_intensity:.1f}, filled={is_filled}")
                
                if is_filled:
                    filled_options.append(option_letter)
            
            question_analysis["filled_options"] = filled_options
            
            # Determine final answer
            if len(filled_options) == 1:
                selected_answers[q_num] = filled_options[0]
                question_analysis["final_answer"] = filled_options[0]
                logger.info(f"🐛 Q{q_num} Final answer: {filled_options[0]}")
            elif len(filled_options) > 1:
                selected_answers[q_num] = "MULTIPLE"
                question_analysis["final_answer"] = "MULTIPLE"
                logger.info(f"🐛 Q{q_num} Multiple selections: {filled_options}")
            else:
                selected_answers[q_num] = "NONE"
                question_analysis["final_answer"] = "NONE"
                logger.info(f"🐛 Q{q_num} No selection detected")
            
            bubble_debug["questions_analyzed"][q_num] = question_analysis
        
        bubble_debug["selected_answers"] = selected_answers
        debug_info["step_4_bubble_analysis"] = bubble_debug
        
        # STEP 6: Final Evaluation
        logger.info("🐛 STEP 6: Final evaluation")
        evaluation_results = evaluate_answers(selected_answers, answer_key_dict)
        
        evaluation_debug = {
            "comparison_details": [],
            "summary": evaluation_results
        }
        
        for q_num, correct_answer in answer_key_dict.items():
            selected = selected_answers.get(q_num, "NONE")
            comparison = {
                "question_number": q_num,
                "correct_answer": correct_answer,
                "detected_answer": selected,
                "is_correct": selected == correct_answer,
                "status": "not_attempted" if selected == "NONE" else 
                         "multiple_selection" if selected == "MULTIPLE" else
                         "correct" if selected == correct_answer else "incorrect"
            }
            evaluation_debug["comparison_details"].append(comparison)
        
        debug_info["step_5_answer_evaluation"] = evaluation_debug
        debug_info["final_results"] = evaluation_results
        
        logger.info(f"🐛 DEBUG COMPLETE: Score = {evaluation_results['score_percentage']:.2f}%")
        
        return JSONResponse(content={
            "debug_mode": True,
            "debug_information": str(debug_info),
            "final_results": str(evaluation_results)
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🐛 DEBUG ERROR: {str(e)}")
        debug_info["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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

