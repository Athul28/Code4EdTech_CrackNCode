import requests
import json
from pprint import pprint

def test_debug_omr_api():
    """Test the DEBUG OMR processing API with detailed output"""
    
    # API endpoint for debug
    url = "http://localhost:8000/debug_omr"
    
    # Sample answer key in CSV format
    answer_key_csv = """question_number,answer
1,a
2,c
3,c
4,c
5,c
6,a
7,c
8,c
9,b
10,c
11,a
12,a
13,d
14,a
15,b
16,a
17,c
18,d
19,a
20,b
21,a
22,d
23,b
24,a
25,c
26,b
27,a
28,b
29,d
30,c
31,c
32,a
33,b
34,c
35,a
36,b
37,d
38,b
39,a
40,b
41,c
42,c
43,c
44,b
45,b
46,a
47,c
48,b
49,d
50,a
51,c
52,b
53,c
54,c
55,a
56,b
57,b
58,a
59,b
60,b
61,b
62,c
63,a
64,b
65,c
66,b
67,b
68,c
69,c
70,b
71,b
72,b
73,d
74,b
75,a
76,b
77,b
78,b
79,b
80,b
81,a
82,b
83,c
84,b
85,c
86,b
87,b
88,b
89,a
90,b
91,c
92,b
93,c
94,b
95,b
96,b
97,c
98,a
99,b
100,c"""
    
    
    # Path to your test image
    image_path = "./data/images/a2449b11-Img1.jpeg"  # Update this path as needed
    
    try:
        # Prepare the request
        with open(image_path, 'rb') as image_file:
            files = {
                'image': ('test_image.jpeg', image_file, 'image/jpeg')
            }
            data = {
                'answer_key': answer_key_csv
            }
            
            # Send the request
            print("🐛 Sending request to DEBUG OMR API...")
            print("=" * 80)
            response = requests.post(url, files=files, data=data)
            
            # Check response
            if response.status_code == 200:
                results = response.json()
                
                print("✅ DEBUG OMR Processing completed successfully!")
                print("=" * 80)
                
                # Print debug information step by step
                debug_info = results.get('debug_information', {})
                
                # Step 1: Image Preprocessing
                print("\n🔍 STEP 1: Image Preprocessing")
                print("-" * 50)
                step1 = debug_info.get('step_1_image_preprocessing', {})
                print(f"Original filename: {step1.get('original_filename')}")
                print(f"File size: {step1.get('file_size_bytes'):,} bytes")
                print(f"Content type: {step1.get('content_type')}")
                print(f"Processed shape: {step1.get('processed_image_shape')}")
                print(f"Data type: {step1.get('processed_image_dtype')}")
                
                # Step 2: Answer Key
                print("\n🔍 STEP 2: Answer Key Processing")
                print("-" * 50)
                step2 = debug_info.get('step_2_answer_key', {})
                print(f"Total questions in key: {step2.get('total_questions_in_key')}")
                print(f"Sample answers: {step2.get('answer_key_sample')}")
                
                # Step 3: Model Predictions
                print("\n🔍 STEP 3: YOLO Model Predictions")
                print("-" * 50)
                step3 = debug_info.get('step_2_model_predictions', {})
                print(f"Confidence threshold: {step3.get('confidence_threshold')}")
                print(f"Total detections: {len(step3.get('detections', []))}")
                
                # Show bounding box visualization info
                vis_path = step3.get('visualization_saved_to')
                if vis_path:
                    print(f"🖼️  Bounding boxes visualization saved to: {vis_path}")
                
                # Show first few detections
                detections = step3.get('detections', [])
                for i, detection in enumerate(detections[:5]):  # Show first 5
                    print(f"  Detection {i+1}: {detection['class_name']} "
                          f"[{detection['bbox']}] conf={detection['confidence']:.3f}")
                
                if len(detections) > 5:
                    print(f"  ... and {len(detections) - 5} more detections")
                
                # Step 4: OCR Processing
                print("\n🔍 STEP 4: OCR Processing")
                print("-" * 50)
                step4 = debug_info.get('step_3_ocr_detections', {})
                print(f"Total detections processed: {step4.get('total_detections_processed')}")
                print(f"Question mappings found: {step4.get('question_number_mappings')}")
                
                # Show detected questions and options summary
                question_summary = step4.get('question_options_summary', {})
                if question_summary:
                    print(f"\n📋 DETECTED QUESTIONS AND OPTIONS:")
                    print("-" * 30)
                    for q_num in sorted(question_summary.keys()):
                        options = question_summary[q_num]
                        options_str = ", ".join([f"{opt['option']}" for opt in options])
                        print(f"  Question {q_num}: [{options_str}] ({len(options)} options)")
                
                # Show OCR results for first few detections
                ocr_results = step4.get('ocr_results', [])
                print(f"\n🔍 OCR Details (first 3):")
                for i, ocr_result in enumerate(ocr_results[:3]):  # Show first 3
                    print(f"  Detection {i+1}:")
                    print(f"    Expanded bbox: {ocr_result.get('expanded_bbox')}")
                    print(f"    Question number: {ocr_result.get('extracted_question_number')}")
                    raw_results = ocr_result.get('ocr_raw_results', [])
                    for raw in raw_results:
                        print(f"    OCR text: '{raw['text']}' (conf: {raw['confidence']:.3f})")
                
                # Step 5: Bubble Analysis
                print("\n🔍 STEP 5: Bubble Analysis")
                print("-" * 50)
                step5 = debug_info.get('step_4_bubble_analysis', {})
                print(f"Intensity threshold: {step5.get('intensity_threshold')}")
                
                questions_analyzed = step5.get('questions_analyzed', {})
                for q_num, analysis in list(questions_analyzed.items())[:5]:  # Show first 5 questions
                    print(f"  Question {q_num}:")
                    print(f"    Total bubbles: {analysis['total_bubbles']}")
                    print(f"    Final answer: {analysis['final_answer']}")
                    
                    for bubble in analysis['bubble_details']:
                        print(f"      Option {bubble['option']}: intensity={bubble['mean_intensity']:.1f}, "
                              f"filled={bubble['is_filled']}")
                
                # Step 6: Final Evaluation
                print("\n🔍 STEP 6: Final Evaluation")
                print("-" * 50)
                final_results = results.get('final_results', {})
                print(f"Total Questions: {final_results.get('total_questions')}")
                print(f"Attempted: {final_results.get('attempted')}")
                print(f"Correct: {final_results.get('correct')}")
                print(f"Incorrect: {final_results.get('incorrect')}")
                print(f"Score: {final_results.get('score_percentage', 0):.2f}%")
                print(f"Not Attempted: {final_results.get('not_attempted')}")
                print(f"Multiple Selections: {final_results.get('multiple_selections')}")
                
                # Show detailed comparison for first few questions
                step6 = debug_info.get('step_5_answer_evaluation', {})
                comparison_details = step6.get('comparison_details', [])
                print(f"\n📝 First 10 Questions Detailed Results:")
                for detail in comparison_details[:10]:
                    status_emoji = "✅" if detail['status'] == 'correct' else \
                                  "❌" if detail['status'] == 'incorrect' else \
                                  "⚪" if detail['status'] == 'not_attempted' else "🔄"
                    print(f"  {status_emoji} Q{detail['question_number']}: "
                          f"Detected='{detail['detected_answer']}' "
                          f"Correct='{detail['correct_answer']}' "
                          f"({detail['status']})")
                
                print("\n" + "=" * 80)
                print("🎯 DEBUG ANALYSIS COMPLETE!")
                print("Check the console logs for even more detailed step-by-step processing information.")
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at {image_path}")
        print("Please update the image_path variable to point to a valid image file.")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API server.")
        print("Make sure the FastAPI server is running at http://localhost:8000")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

def save_debug_output_to_file():
    """Save the complete debug output to a JSON file for detailed analysis"""
    
    url = "http://localhost:8000/debug_omr"
    
    answer_key_csv = """question_number,answer
1,A
2,B
3,C
4,D
5,A"""
    
    image_path = "./data/images/a2449b11-Img1.jpeg"
    
    try:
        with open(image_path, 'rb') as image_file:
            files = {'image': ('test_image.jpeg', image_file, 'image/jpeg')}
            data = {'answer_key': answer_key_csv}
            
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                results = response.json()
                
                # Save to file
                output_file = "debug_output.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"✅ Complete debug output saved to: {output_file}")
                print("You can analyze this file for detailed debugging information.")
                
            else:
                print(f"❌ Error saving debug output: {response.status_code}")
                
    except Exception as e:
        print(f"❌ Error saving debug output: {str(e)}")

if __name__ == "__main__":
    print("🐛 DEBUG OMR API TESTER")
    print("=" * 80)
    
    # Test the debug endpoint
    test_debug_omr_api()
    
    print("\n" + "=" * 80)
    print("💾 Saving complete debug output to file...")
    save_debug_output_to_file()
    
    print("\n" + "=" * 80)
    print("🔧 Debug Testing Complete!")
    print("Use this detailed output to:")
    print("1. Verify model predictions are accurate")
    print("2. Check OCR text extraction quality")
    print("3. Analyze bubble fill detection thresholds")
    print("4. Tune parameters for better accuracy")