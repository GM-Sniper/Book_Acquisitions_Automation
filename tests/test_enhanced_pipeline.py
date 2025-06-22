import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vision.preprocessing import preprocess_image
from src.vision.OCR_Processing import extract_text_with_confidence
from src.utils.isbn_detection import extract_and_validate_isbns
from src.metadata.metadata_extraction import extract_metadata_with_gemini, metadata_combiner
import json
import time

def test_enhanced_pipeline():
    """Test the enhanced pipeline using existing code with new features"""
    
    print("Testing Enhanced Pipeline (Existing Code + New Features)")
    print("=" * 60)
    
    # Test with all book images
    test_images = [
        "data/raw_images/book1_front.jpg",
        "data/raw_images/book1_back.jpg",
        "data/raw_images/book2_front.jpg", 
        "data/raw_images/book2_back.jpg",
        "data/raw_images/book3_front.jpg",
        "data/raw_images/book3_back.jpg"
    ]
    
    results = []
    
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
    
    # Generate summary
    print(f"\n{'='*60}")
    print("ENHANCED PIPELINE TEST SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"Total images processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        avg_confidence = sum(r['text_confidence'] for r in successful) / len(successful)
        avg_time = sum(r['processing_time'] for r in successful) / len(successful)
        total_isbns = sum(r['isbn_results']['total_found'] for r in successful)
        validated_isbns = sum(r['isbn_results']['validated_count'] for r in successful)
        
        print(f"Average text confidence: {avg_confidence:.2f}")
        print(f"Average processing time: {avg_time:.2f} seconds")
        print(f"Total ISBNs found: {total_isbns}")
        print(f"Total ISBNs validated: {validated_isbns}")
    
    # Save detailed results
    os.makedirs("data/test_results", exist_ok=True)
    report_path = f"data/test_results/enhanced_pipeline_test_{int(time.time())}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {report_path}")
    
    return results

if __name__ == "__main__":
    test_enhanced_pipeline() 