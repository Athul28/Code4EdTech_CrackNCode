import requests
import json

def test_omr_api():
    """Test the OMR processing API"""
    
    # API endpoint
    url = "http://localhost:8000/predict"
    
    # Sample answer key in CSV format
    answer_key_csv = """question_number,answer
1,['a']
2,['c']
3,['c']
4,['c']
5,['c']
6,['a']
7,['c']
8,['c']
9,['b']
10,['c']
11,['a']
12,['a']
13,['d']
14,['a']
15,['b']
16,['a']
17,['c']
18,['d']
19,['a']
20,['b']
21,['a']
22,['d']
23,['b']
24,['a']
25,['c']
26,['b']
27,['a']
28,['b']
29,['d']
30,['c']
31,['c']
32,['a']
33,['b']
34,['c']
35,['a']
36,['b']
37,['d']
38,['b']
39,['a']
40,['b']
41,['c']
42,['c']
43,['c']
44,['b']
45,['b']
46,['a']
47,['c']
48,['b']
49,['d']
50,['a']
51,['c']
52,['b']
53,['c']
54,['c']
55,['a']
56,['b']
57,['b']
58,['a']
59,['b']
60,['b']
61,['b']
62,['c']
63,['a']
64,['b']
65,['c']
66,['b']
67,['b']
68,['c']
69,['c']
70,['b']
71,['b']
72,['b']
73,['d']
74,['b']
75,['a']
76,['b']
77,['b']
78,['b']
79,['b']
80,['b']
81,['a']
82,['b']
83,['c']
84,['b']
85,['c']
86,['b']
87,['b']
88,['b']
89,['a']
90,['b']
91,['c']
92,['b']
93,['c']
94,['b']
95,['b']
96,['b']
97,['c']
98,['a']
99,['b']
100,['c']"""
    
    # Path to your test image
    image_path = "./data/images/b9716df6-Img4.jpeg"  # Update this path as needed
    
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
            print("Sending request to OMR API...")
            response = requests.post(url, files=files, data=data)
            
            # Check response
            if response.status_code == 200:
                results = response.json()
                print(results)


                return
                print("\n✅ SUCCESS! OMR Processing completed.")
                print(f"📊 Results Summary:")
                print(f"   Total Questions: {results['total_questions']}")
                print(f"   Attempted: {results['attempted']}")
                print(f"   Correct: {results['correct']}")
                print(f"   Incorrect: {results['incorrect']}")
                print(f"   Score: {results['score_percentage']:.2f}%")
                print(f"   Not Attempted: {results['not_attempted']}")
                print(f"   Multiple Selections: {results['multiple_selections']}")
                
                # Show detailed results for first few questions
                print(f"\n📝 First 5 Questions Details:")
                for detail in results['detailed_results'][:5]:
                    print(f"   Q{detail['question_number']}: {detail['selected_answer']} (Correct: {detail['correct_answer']}) - {detail['status']}")
                
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

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            health_data = response.json()
            print("🏥 Health Check:")
            print(f"   Status: {health_data['status']}")
            print(f"   Model Loaded: {health_data['model_loaded']}")
            # print(f"   OCR Ready: {health_data['ocr_ready']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API server for health check.")

if __name__ == "__main__":
    print("🚀 Testing OMR API")
    print("=" * 50)
    
    # Test health check first
    test_health_check()
    print()
    
    # Test the main OMR processing
    test_omr_api()
    
    print("\n" + "=" * 50)
    print("🎯 To test with your own images:")
    print("1. Update the 'image_path' variable in this script")
    print("2. Modify the 'answer_key_csv' with your correct answers")
    print("3. Run this script again")
