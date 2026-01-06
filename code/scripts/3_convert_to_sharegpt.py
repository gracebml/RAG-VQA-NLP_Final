"""
Step 3: Convert VQA data to ShareGPT format for fine-tuning

Input:
    - data/vqa_train.json
    - data/vqa_test.json

Output:
    - llama-dataset/vqa_vietnamese_train.json (ShareGPT format)
    - llama-dataset/vqa_vietnamese_test.json (ShareGPT format)

Usage:
    python 3_convert_to_sharegpt.py
"""
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path for importing prompts
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from prompts import (
        SYSTEM_PROMPT, build_user_prompt,
        NO_CONTEXT_MESSAGE, NO_OCR_MESSAGE, NO_CAPTION_MESSAGE
    )
    USE_CENTRALIZED_PROMPTS = True
except ImportError:
    USE_CENTRALIZED_PROMPTS = False
    print("Warning: Could not import centralized prompts, using inline prompts")


def load_json(file_path: Path) -> list:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_context_as_rag(cultural_significance, historical_context) -> str:
    """
    Format cultural_significance và historical_context như kiến thức bổ sung
    Gộp tất cả thành một khối kiến thức duy nhất
    Handle cả string và list
    """
    context_parts = []
    
    # Handle cultural_significance (có thể là string hoặc list)
    if cultural_significance:
        if isinstance(cultural_significance, list):
            context_parts.extend(cultural_significance)
        else:
            context_parts.append(cultural_significance)
    
    # Handle historical_context (có thể là string hoặc list)
    if historical_context:
        if isinstance(historical_context, list):
            context_parts.extend(historical_context)
        else:
            context_parts.append(historical_context)
    
    if not context_parts:
        if USE_CENTRALIZED_PROMPTS:
            return NO_CONTEXT_MESSAGE
        return "Không có thông tin liên quan."
    
    return "\n\n".join(context_parts)


def create_training_example(vqa_item: dict) -> dict:
    """
    Create training example in ShareGPT format for LLaMA Factory
    
    Format GIỐNG HỆT như trong answering.py generate_answer()
    Uses centralized prompts from src/prompts.py for consistency.
    """
    # Extract data
    image_path = vqa_item["image_path"]
    image_filename = Path(image_path).name
    
    question = vqa_item["question"]
    
    # Use centralized constants if available
    if USE_CENTRALIZED_PROMPTS:
        caption = vqa_item.get("caption", NO_CAPTION_MESSAGE) or NO_CAPTION_MESSAGE
        ocr = vqa_item.get("ocr", NO_OCR_MESSAGE) or NO_OCR_MESSAGE
    else:
        caption = vqa_item.get("caption", "Không có mô tả hình ảnh.") or "Không có mô tả hình ảnh."
        ocr = vqa_item.get("ocr", "Không có văn bản.") or "Không có văn bản."
    
    # Extract answer components from FLAT structure (after preprocessing)
    # answer is already merged (answer + detail_explaination)
    answer = vqa_item.get("answer", "")
    cultural_significance = vqa_item.get("cultural_significance", "")
    historical_context = vqa_item.get("historical_context", "")
    
    # Format context as RAG passages
    context = format_context_as_rag(cultural_significance, historical_context)
    
    # Build prompt using centralized function if available
    if USE_CENTRALIZED_PROMPTS:
        user_prompt = build_user_prompt(
            caption=caption,
            ocr_text=ocr,
            context=context,
            question=question
        )
        system_prompt = SYSTEM_PROMPT
    else:
        # Fallback to inline prompts (should match prompts.py)
        user_prompt = (
            f"Dựa vào mô tả hình ảnh: {caption}\n\n"
            f"Thông tin văn bản trong ảnh: {ocr}\n\n"
            f"Kiến thức lịch sử và văn hóa:\n{context}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Hãy trả lời bằng tiếng Việt chuẩn (KHÔNG dùng tiếng Trung hay ngôn ngữ khác). "
            "Nếu phù hợp, hãy cung cấp giải thích chi tiết, ý nghĩa văn hóa và bối cảnh lịch sử."
        )
        system_prompt = (
            "Bạn là trợ lý VQA chuyên về lịch sử và văn hóa Việt Nam. "
            "Trả lời câu hỏi dựa trên HÌNH ẢNH là chính. "
            "Mô tả hình ảnh và văn bản OCR chỉ là thông tin tham khảo, có thể không chính xác. "
            "Kiến thức bổ sung giúp giải thích sâu hơn khi cần.\n\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. CHỈ trả lời bằng tiếng Việt thuần túy, TUYỆT ĐỐI KHÔNG dùng tiếng Trung, Hàn, Nhật hay bất kỳ ngôn ngữ nào khác.\n"
            "2. Sử dụng chính tả tiếng Việt chuẩn với dấu thanh đúng.\n"
            "3. Trả lời rõ ràng, có cấu trúc và dễ hiểu."
        )
    
    # Answer already merged in preprocessing step
    assistant_response = answer
    
    # ShareGPT format for LLaMA Factory
    return {
        "conversations": [
            {
                "from": "system",
                "value": system_prompt
            },
            {
                "from": "human",
                "value": f"<image>\n\n{user_prompt}"
            },
            {
                "from": "gpt",
                "value": assistant_response
            }
        ],
        "images": [image_filename]
    }


def convert_dataset(input_file: Path, output_file: Path, dataset_name: str):
    """Convert VQA dataset to ShareGPT format"""
    
    print(f"\n Loading {dataset_name} from {input_file}...")
    data = load_json(input_file)
    print(f" Loaded {len(data)} samples")
    
    print(f"\n Converting to ShareGPT format...")
    converted_data = []
    
    for item in tqdm(data, desc=f"Converting {dataset_name}"):
        try:
            example = create_training_example(item)
            converted_data.append(example)
        except Exception as e:
            print(f"\n  Error converting {item.get('image_path', 'unknown')}: {e}")
    
    print(f" Converted {len(converted_data)} samples")
    
    # Save
    print(f"\n Saving to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    print(f" Saved {len(converted_data)} samples")
    
    return converted_data


def main():
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "llama-dataset"
    
    INPUT_TRAIN = DATA_DIR / "vqa_train.json"
    INPUT_TEST = DATA_DIR / "vqa_test.json"
    
    OUTPUT_TRAIN = OUTPUT_DIR / "vqa_vietnamese_train.json"
    OUTPUT_TEST = OUTPUT_DIR / "vqa_vietnamese_test.json"
    
    print("="*80)
    print("STEP 3: CONVERT TO SHAREGPT FORMAT")
    print("="*80)
    
    # Convert train
    train_data = convert_dataset(INPUT_TRAIN, OUTPUT_TRAIN, "train")
    
    # Convert test
    test_data = convert_dataset(INPUT_TEST, OUTPUT_TEST, "test")
    
    # Display sample
    print("\n" + "="*80)
    print("SAMPLE SHAREGPT DATA")
    print("="*80)
    
    sample = train_data[0]
    print(f"\n Image: {sample['images'][0]}")
    print(f"\n System: {sample['conversations'][0]['value'][:100]}...")
    print(f"\n Human:\n{sample['conversations'][1]['value'][:300]}...")
    print(f"\n Assistant:\n{sample['conversations'][2]['value'][:200]}...")
    
    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    import statistics
    
    # Human prompt lengths
    human_lengths = [len(ex['conversations'][1]['value']) for ex in train_data]
    
    # Assistant response lengths
    assistant_lengths = [len(ex['conversations'][2]['value']) for ex in train_data]
    
    print(f"\nTrain set ({len(train_data)} samples):")
    print(f"  Human prompts:")
    print(f"    Min: {min(human_lengths)} chars")
    print(f"    Max: {max(human_lengths)} chars")
    print(f"    Mean: {statistics.mean(human_lengths):.1f} chars")
    
    print(f"  Assistant responses:")
    print(f"    Min: {min(assistant_lengths)} chars")
    print(f"    Max: {max(assistant_lengths)} chars")
    print(f"    Mean: {statistics.mean(assistant_lengths):.1f} chars")
    
    print(f"\nTest set: {len(test_data)} samples")
    
    print("\n" + "="*80)
    print(" STEP 3 COMPLETED!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  Train: {OUTPUT_TRAIN}")
    print(f"  Test: {OUTPUT_TEST}")
    print("\n All data preparation steps completed!")
    print("\nNext steps:")
    print("1. Update finetuning/dataset_info.json")
    print("2. Run fine-tuning with LLaMA Factory")


if __name__ == "__main__":
    main()
