"""
Step 1: Merge captions and OCR into VQA dataset

Input:
    - data/vqa.json (VQA ground truth)
    - data/raw/image_captions.json (Generated captions)
    - data/raw/image_ocr.json (OCR text)

Output:
    - data/vqa_with_metadata.json (VQA + caption + ocr)

Usage:
    python 1_merge_caption_ocr.py
"""
import json
from pathlib import Path
from tqdm import tqdm


def load_json(file_path: Path) -> dict:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = BASE_DIR / "data" / "raw"
    
    VQA_JSON = DATA_DIR / "vqa_preprocessed.json"  # Use preprocessed version
    CAPTIONS_JSON = RAW_DATA_DIR / "image_captions.json"
    OCR_JSON = RAW_DATA_DIR / "image_ocr.json"
    
    OUTPUT_JSON = DATA_DIR / "vqa_with_metadata.json"
    
    print("="*80)
    print("STEP 1: MERGE CAPTION AND OCR INTO VQA DATASET")
    print("="*80)
    
    # Load data
    print("\n Loading data...")
    print(f"  VQA: {VQA_JSON}")
    print(f"  Captions: {CAPTIONS_JSON}")
    print(f"  OCR: {OCR_JSON}")
    
    vqa_data = load_json(VQA_JSON)
    
    # Load captions - convert from list to dict keyed by image_id
    if CAPTIONS_JSON.exists():
        captions_list = load_json(CAPTIONS_JSON)
        # Convert list to dict: {image_id: {image_id, caption, ...}}
        captions = {item['image_id']: item for item in captions_list}
    else:
        print(f"  Warning: {CAPTIONS_JSON} not found")
        captions = {}
    
    # Load OCR - convert from list to dict keyed by image_id
    if OCR_JSON.exists():
        ocr_list = load_json(OCR_JSON)
        # Convert list to dict: {image_id: {image_id, ocr, ...}}
        ocr_data = {item['image_id']: item for item in ocr_list}
    else:
        print(f"  Warning: {OCR_JSON} not found")
        ocr_data = {}
    
    print(f"\n Loaded:")
    print(f"  VQA entries: {len(vqa_data)}")
    print(f"  Captions: {len(captions)}")
    print(f"  OCR: {len(ocr_data)}")
    
    # Merge data
    print("\n Merging data...")
    merged_data = []
    missing_captions = []
    missing_ocr = []
    
    for item in tqdm(vqa_data, desc="Merging"):
        image_path = item["image_path"]
        # Extract image filename from path (e.g., "images_flat/000075.png" -> "000075.png")
        image_name = Path(image_path).name
        
        # Create merged item (copy original)
        merged_item = item.copy()
        
        # Add caption
        if image_name in captions:
            caption = captions[image_name].get("caption", "").strip()
            if caption:
                merged_item["caption"] = caption
            else:
                missing_captions.append(image_name)
                merged_item["caption"] = ""
        else:
            missing_captions.append(image_name)
            merged_item["caption"] = ""
        
        # Add OCR
        if image_name in ocr_data:
            ocr = ocr_data[image_name].get("ocr", "").strip()
            merged_item["ocr"] = ocr if ocr else ""
        else:
            missing_ocr.append(image_name)
            merged_item["ocr"] = ""
        
        merged_data.append(merged_item)
    
    print(f"\n Merged {len(merged_data)} entries")
    
    if missing_captions:
        print(f"\n  Missing captions: {len(missing_captions)} images")
        print(f"   Sample: {missing_captions[:5]}")
    
    if missing_ocr:
        print(f"\n  Missing OCR: {len(missing_ocr)} images")
        print(f"   Sample: {missing_ocr[:5]}")
    
    # Save
    print(f"\n Saving to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f" Saved {len(merged_data)} entries")
    
    # Display sample
    print("\n" + "="*80)
    print("SAMPLE MERGED DATA")
    print("="*80)
    sample = merged_data[0]
    print(f"\nImage: {sample['image_path']}")
    print(f"Question: {sample['question']}")
    print(f"Caption: {sample['caption'][:100]}..." if len(sample['caption']) > 100 else f"Caption: {sample['caption']}")
    print(f"OCR: {sample['ocr']}")
    # answer is already a merged string after preprocessing
    print(f"Answer: {sample.get('answer', '')}")
    
    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    has_caption = sum(1 for item in merged_data if item.get('caption'))
    has_ocr = sum(1 for item in merged_data if item.get('ocr'))
    
    print(f"\nCoverage:")
    print(f"  Has caption: {has_caption}/{len(merged_data)} ({has_caption/len(merged_data)*100:.1f}%)")
    print(f"  Has OCR: {has_ocr}/{len(merged_data)} ({has_ocr/len(merged_data)*100:.1f}%)")
    
    print("\n" + "="*80)
    print(" STEP 1 COMPLETED!")
    print("="*80)
    print(f"\nOutput: {OUTPUT_JSON}")
    print("\nNext step: Run 2_split_train_test.py")


if __name__ == "__main__":
    main()
