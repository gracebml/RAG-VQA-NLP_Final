"""
Step 2: Shuffle and split data into train and test sets

Input:
    - data/vqa_with_metadata.json

Output:
    - data/vqa_train.json (training set)
    - data/vqa_test.json (test set, ~60 samples)

Usage:
    python 2_split_train_test.py
"""
import json
import random
from pathlib import Path


def load_json(file_path: Path) -> list:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    
    INPUT_JSON = DATA_DIR / "vqa_with_metadata.json"
    OUTPUT_TRAIN = DATA_DIR / "vqa_train.json"
    OUTPUT_TEST = DATA_DIR / "vqa_test.json"
    
    # Configuration
    TEST_SIZE = 60
    RANDOM_SEED = 42
    
    print("="*80)
    print("STEP 2: SHUFFLE AND SPLIT TRAIN/TEST")
    print("="*80)
    
    # Load data
    print(f"\n Loading data from {INPUT_JSON}...")
    data = load_json(INPUT_JSON)
    print(f" Loaded {len(data)} samples")
    
    # Shuffle data
    print(f"\n Shuffling data (seed={RANDOM_SEED})...")
    random.seed(RANDOM_SEED)
    random.shuffle(data)
    print(" Data shuffled")
    
    # Split train/test
    print(f"\n  Splitting data...")
    test_size = min(TEST_SIZE, len(data))
    train_data = data[:-test_size] if len(data) > test_size else []
    test_data = data[-test_size:]
    
    print(f"  Train: {len(train_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # Save train
    print(f"\n Saving train data to {OUTPUT_TRAIN}...")
    with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f" Saved {len(train_data)} training samples")
    
    # Save test
    print(f"\n Saving test data to {OUTPUT_TEST}...")
    with open(OUTPUT_TEST, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f" Saved {len(test_data)} test samples")
    
    # Display samples
    print("\n" + "="*80)
    print("SAMPLE DATA")
    print("="*80)
    
    if train_data:
        print("\n Train sample:")
        sample = train_data[0]
        print(f"  Image: {sample['image_path']}")
        print(f"  Question: {sample['question'][:80]}...")
        print(f"  Has caption: {'Yes' if sample.get('caption') else 'No'}")
        print(f"  Has OCR: {'Yes' if sample.get('ocr') else 'No'}")
    
    if test_data:
        print("\n Test sample:")
        sample = test_data[0]
        print(f"  Image: {sample['image_path']}")
        print(f"  Question: {sample['question'][:80]}...")
        print(f"  Has caption: {'Yes' if sample.get('caption') else 'No'}")
        print(f"  Has OCR: {'Yes' if sample.get('ocr') else 'No'}")
    
    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    print(f"\nDataset split:")
    print(f"  Total: {len(data)} samples")
    print(f"  Train: {len(train_data)} samples ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Test: {len(test_data)} samples ({len(test_data)/len(data)*100:.1f}%)")
    
    print("\n" + "="*80)
    print(" STEP 2 COMPLETED!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  Train: {OUTPUT_TRAIN}")
    print(f"  Test: {OUTPUT_TEST}")
    print("\nNext step: Run 3_convert_to_sharegpt.py")


if __name__ == "__main__":
    main()
