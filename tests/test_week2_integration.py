import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vision.camera_handler import CameraHandler
from src.vision.preprocessing import preprocess_image
from src.vision.OCR_Processing import extract_text_with_confidence
from src.utils.isbn_detection import extract_and_validate_isbns
from src.metadata.metadata_extraction import extract_metadata_with_gemini, metadata_combiner
import json
import time

class Week2Tester:
    def __init__(self):
        self.camera = CameraHandler()
        
        # Create results directory
        os.makedirs("data/test_results", exist_ok=True)
    
    def test_enhanced_pipeline(self, test_images_dir="data/raw_images"):
        """Test the enhanced pipeline using existing code with new features"""
        results = []
        
        # Test with all book images
        test_images = [
            "data/raw_images/book1_front.jpg",
            "data/raw_images/book1_back.jpg",
            "data/raw_images/book2_front.jpg", 
            "data/raw_images/book2_back.jpg",
            "data/raw_images/book3_front.jpg",
            "data/raw_images/book3_back.jpg"
        ]
        
        print(f"Testing enhanced pipeline with {len(test_images)} images...")
        
        for image_path in test_images:
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
                
            print(f"\nProcessing: {os.path.basename(image_path)}")
            start_time = time.time()
            
            try:
                # Step 1: Load and preprocess image
                print("  → Loading and preprocessing image...")
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                
                # Use existing preprocessing with new CLAHE enhancement
                processed_img = preprocess_image(image_bytes)
                print("    ✓ Image preprocessed with CLAHE enhancement")
                
                # Step 2: Extract text with confidence
                print("  → Extracting text with confidence...")
                text_result = extract_text_with_confidence(image_path)
                print(f"    ✓ Text extracted (confidence: {text_result['confidence']:.2f})")
                print(f"    ✓ Word count: {text_result['word_count']}")
                
                # Step 3: Extract and validate ISBNs
                print("  → Extracting and validating ISBNs...")
                isbn_results = extract_and_validate_isbns(text_result['text'])
                print(f"    ✓ Found {isbn_results['total_found']} ISBNs")
                print(f"    ✓ Validated {isbn_results['validated_count']} ISBNs")
                
                # Show validation results
                for isbn_data in isbn_results['isbns']:
                    if isbn_data['validation']['valid']:
                        print(f"      ✓ {isbn_data['isbn']}: {isbn_data['validation']['title']}")
                    else:
                        print(f"      ✗ {isbn_data['isbn']}: {isbn_data['validation']['error']}")
                
                # Step 4: Extract metadata with Gemini
                print("  → Extracting metadata with Gemini...")
                metadata = extract_metadata_with_gemini(text_result['text'])
                if metadata:
                    print(f"    ✓ Title: {metadata.get('title', 'Not found')}")
                    print(f"    ✓ Authors: {metadata.get('authors', 'Not found')}")
                else:
                    print("    ✗ Metadata extraction failed")
                
                # Step 5: Combine results
                isbn_list = [isbn['isbn'] for isbn in isbn_results['isbns']]
                combined_metadata = metadata_combiner(metadata or {}, isbn_list)
                
                processing_time = time.time() - start_time
                
                result = {
                    'image_file': os.path.basename(image_path),
                    'processing_time': processing_time,
                    'text_confidence': text_result['confidence'],
                    'word_count': text_result['word_count'],
                    'isbn_results': isbn_results,
                    'metadata': combined_metadata,
                    'success': True
                }
                
                print(f"  ✓ Processing completed in {processing_time:.2f} seconds")
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                result = {
                    'image_file': os.path.basename(image_path),
                    'success': False,
                    'error': str(e)
                }
            
            results.append(result)
        
        # Generate summary report
        self._generate_test_report(results)
        return results
    
    def test_camera_capture(self, num_books=3):
        """Test camera capture functionality"""
        print(f"Starting camera capture test for {num_books} books...")
        print("Make sure you have books ready!")
        
        captures = self.camera.batch_capture_mode(num_books)
        
        if captures:
            print(f"\nSuccessfully captured {len(captures)} books")
            
            # Process all captured images
            all_results = []
            for capture in captures:
                for image_path in capture['images']:
                    if os.path.exists(image_path):
                        print(f"Processing captured image: {image_path}")
                        # Process this image through the enhanced pipeline
                        # (This would call test_enhanced_pipeline for just this image)
            
            return captures
        
        return []
    
    def _generate_test_report(self, results):
        """Generate comprehensive test report"""
        report = {
            'test_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_images': len(results),
            'successful_processing': len([r for r in results if r.get('success', False)]),
            'failed_processing': len([r for r in results if not r.get('success', False)]),
            'avg_confidence': sum([r.get('text_confidence', 0) for r in results if r.get('success', False)]) / len([r for r in results if r.get('success', False)]) if any(r.get('success', False) for r in results) else 0,
            'avg_processing_time': sum([r.get('processing_time', 0) for r in results]) / len(results) if results else 0,
            'total_isbns_found': sum([r.get('isbn_results', {}).get('total_found', 0) for r in results if r.get('success', False)]),
            'total_isbns_validated': sum([r.get('isbn_results', {}).get('validated_count', 0) for r in results if r.get('success', False)]),
            'results': results
        }
        
        # Save detailed report
        report_path = f"data/test_results/week2_enhanced_test_report_{int(time.time())}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n{'='*50}")
        print("WEEK 2 ENHANCED TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Total images processed: {report['total_images']}")
        print(f"Successful: {report['successful_processing']}")
        print(f"Failed: {report['failed_processing']}")
        print(f"Average text confidence: {report['avg_confidence']:.2f}")
        print(f"Average processing time: {report['avg_processing_time']:.2f} seconds")
        print(f"Total ISBNs found: {report['total_isbns_found']}")
        print(f"Total ISBNs validated: {report['total_isbns_validated']}")
        print(f"Detailed report saved: {report_path}")
        
        return report

def run_week2_tests():
    """Run all Week 2 tests"""
    tester = Week2Tester()
    
    print("Week 2 Enhanced Integration Testing")
    print("===================================")
    
    # Test 1: Process existing images with enhanced pipeline
    if os.path.exists("data/raw_images") and os.listdir("data/raw_images"):
        print("\n1. Testing enhanced pipeline with all book images...")
        tester.test_enhanced_pipeline()
    else:
        print("\n1. No existing images found. Skipping pipeline test.")
        print("   Add some book images to data/raw_images/ first.")
    
    # Test 2: Interactive capture test (optional)
    choice = input("\n2. Test camera capture mode? (y/n): ").lower()
    if choice == 'y':
        num_books = int(input("How many books to capture? (1-5): "))
        tester.test_camera_capture(num_books)
    
    print("\nWeek 2 enhanced testing complete!")

if __name__ == "__main__":
    run_week2_tests() 