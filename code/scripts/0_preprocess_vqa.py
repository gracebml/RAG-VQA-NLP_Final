"""
Step 0: Preprocess VQA dataset - Restructure answer fields

Thực hiện:
1. Chuyển historical_context và cultural_significance từ trong answer ra ngoài
2. Gộp answer và detail_explaination thành một trường answer duy nhất

Input:
    - data/raw/vqa.json (original format)

Output:
    - data/vqa_preprocessed.json (restructured format)
    - Backup: data/vqa_original.json

Usage:
    python 0_preprocess_vqa.py
"""
import json
import shutil
from pathlib import Path
from tqdm import tqdm


def load_json(file_path: Path) -> list:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def preprocess_item(item: dict) -> dict:
    """
    Restructure a single VQA item:
    
    Before:
    {
        "answer": {
            "answer": "...",
            "detail_explaination": "...",
            "cultural_significance": "...",
            "historical_context": "..."
        }
    }
    
    After:
    {
        "answer": "answer + detail_explaination",
        "cultural_significance": "...",
        "historical_context": "..."
    }
    """
    # Copy original item
    new_item = item.copy()
    
    # Get answer object
    answer_obj = item.get("answer", {})
    
    # Extract fields
    answer_text = answer_obj.get("answer", "")
    detail_explanation = answer_obj.get("detail_explaination", "")  # Note: typo in original
    cultural_significance = answer_obj.get("cultural_significance", "")
    historical_context = answer_obj.get("historical_context", "")
    
    # Merge answer and detail_explaination
    merged_answer = answer_text
    if detail_explanation:
        merged_answer += f"\n\n{detail_explanation}"
    
    # Update item
    new_item["answer"] = merged_answer
    new_item["cultural_significance"] = cultural_significance
    new_item["historical_context"] = historical_context
    
    return new_item


def main():
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    
    INPUT_JSON = DATA_DIR / "raw" / "vqa.json"
    OUTPUT_JSON = DATA_DIR / "vqa_preprocessed.json"
    BACKUP_JSON = DATA_DIR / "vqa_original.json"
    
    print("="*80)
    print("STEP 0: PREPROCESS VQA DATASET")
    print("="*80)
    
    # Check input file
    if not INPUT_JSON.exists():
        print(f"\n Error: {INPUT_JSON} not found!")
        return
    
    # Backup original file
    print(f"\n Creating backup: {BACKUP_JSON}")
    shutil.copy2(INPUT_JSON, BACKUP_JSON)
    print(" Backup created")
    
    # Load data
    print(f"\n Loading data from {INPUT_JSON}...")
    data = load_json(INPUT_JSON)
    print(f" Loaded {len(data)} samples")
    
    # Display original format
    print("\n" + "="*80)
    print("ORIGINAL FORMAT (sample)")
    print("="*80)
    sample = data[0]
    print(f"\nImage: {sample['image_path']}")
    print(f"Question: {sample['question'][:80]}...")
    print(f"\nAnswer object keys: {list(sample['answer'].keys())}")
    print(f"  answer: {sample['answer']['answer'][:100]}...")
    if sample['answer'].get('detail_explaination'):
        print(f"  detail_explaination: {sample['answer']['detail_explaination'][:100]}...")
    if sample['answer'].get('cultural_significance'):
        print(f"  cultural_significance: {str(sample['answer']['cultural_significance'])[:100]}...")
    if sample['answer'].get('historical_context'):
        print(f"  historical_context: {str(sample['answer']['historical_context'])[:100]}...")
    
    # Preprocess data
    print(f"\n Preprocessing {len(data)} samples...")
    preprocessed_data = []
    
    for item in tqdm(data, desc="Preprocessing"):
        try:
            new_item = preprocess_item(item)
            preprocessed_data.append(new_item)
        except Exception as e:
            print(f"\n Warning: Error processing {item.get('image_path', 'unknown')}: {e}")
    
    print(f" Preprocessed {len(preprocessed_data)} samples")
    
    # Save
    print(f"\n Saving to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(preprocessed_data, f, ensure_ascii=False, indent=2)
    print(f" Saved {len(preprocessed_data)} samples")
    
    # Display new format
    print("\n" + "="*80)
    print("NEW FORMAT (sample)")
    print("="*80)
    new_sample = preprocessed_data[0]
    print(f"\nImage: {new_sample['image_path']}")
    print(f"Question: {new_sample['question'][:80]}...")
    print(f"\nTop-level keys: {list(new_sample.keys())}")
    print(f"\nanswer (merged): {new_sample['answer'][:150]}...")
    if new_sample.get('cultural_significance'):
        print(f"\ncultural_significance: {str(new_sample['cultural_significance'])[:100]}...")
    if new_sample.get('historical_context'):
        print(f"\nhistorical_context: {str(new_sample['historical_context'])[:100]}...")
    
    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    has_cultural = sum(1 for item in preprocessed_data if item.get('cultural_significance'))
    has_historical = sum(1 for item in preprocessed_data if item.get('historical_context'))
    
    print(f"\nData coverage:")
    print(f"  Has cultural_significance: {has_cultural}/{len(preprocessed_data)} ({has_cultural/len(preprocessed_data)*100:.1f}%)")
    print(f"  Has historical_context: {has_historical}/{len(preprocessed_data)} ({has_historical/len(preprocessed_data)*100:.1f}%)")
    
    # Answer length statistics
    answer_lengths = [len(item['answer']) for item in preprocessed_data]
    import statistics
    print(f"\nMerged answer lengths:")
    print(f"  Min: {min(answer_lengths)} chars")
    print(f"  Max: {max(answer_lengths)} chars")
    print(f"  Mean: {statistics.mean(answer_lengths):.1f} chars")
    print(f"  Median: {statistics.median(answer_lengths):.1f} chars")
    
    print("\n" + "="*80)
    print(" STEP 0 COMPLETED!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  Preprocessed: {OUTPUT_JSON}")
    print(f"  Backup: {BACKUP_JSON}")
    print("\n IMPORTANT: Update Step 1 to use vqa_preprocessed.json instead of vqa.json")
    print("\nNext step: Run 1_merge_caption_ocr.py")


if __name__ == "__main__":
    main()
