#!/usr/bin/env python3
"""
Demo script để chạy RAG-VQA pipeline trên 1 ví dụ mẫu
Sử dụng cho LOCAL testing

Usage:
    python run_demo.py
    python run_demo.py --image path/to/image.jpg --question "Câu hỏi của bạn?"
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import RAGVQAPipeline
from PIL import Image
import json
import os


def get_default_answering_model():
    """Tìm fine-tuned model, fallback về base model nếu không có"""
    # Các path có thể có fine-tuned model
    possible_paths = [
        "../models/qwen2vl-7b-vqa-grounded",  # Model chính
        "../models/finetuned/qwen2vl-2b-vqa",
        "../models/qwen2vl-2b-vqa-finetuned",
        "../models/finetuned",
        "/kaggle/input/finetuned-model"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"    Found fine-tuned model: {path}")
            return path
    
    print("   Fine-tuned model not found, using base model")
    return "Qwen/Qwen2-VL-2B-Instruct"


def main():
    parser = argparse.ArgumentParser(description="Demo RAG-VQA Pipeline")
    parser.add_argument(
        "--image", 
        type=str, 
        default="../data/data-benchmark/benchmark_images/000000.jpg",
        help="Đường dẫn đến hình ảnh"
    )
    parser.add_argument(
        "--question", 
        type=str, 
        default="Đây là công trình kiến trúc gì? Được xây dựng khi nào?",
        help="Câu hỏi về hình ảnh"
    )
    parser.add_argument(
        "--use-4bit", 
        action="store_true", 
        default=True,
        help="Sử dụng 4-bit quantization (khuyến nghị)"
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Vision model name"
    )
    parser.add_argument(
        "--answering-model",
        type=str,
        default=None,
        help="Answering model name hoặc path (mặc định: tự động tìm fine-tuned model)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RAG-VQA DEMO - Vietnamese Cultural Visual Question Answering")
    print("=" * 80)
    print()
    
    # Auto-detect fine-tuned model if not specified
    if args.answering_model is None:
        args.answering_model = get_default_answering_model()
    
    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f" Lỗi: Không tìm thấy hình ảnh tại {image_path}")
        print(f"   Vui lòng kiểm tra đường dẫn hoặc sử dụng mặc định")
        return
    
    print(f" Image: {image_path}")
    print(f" Question: {args.question}")
    print()
    
    # Initialize pipeline
    print(" Initializing RAG-VQA Pipeline...")
    print(f"   Vision Model: {args.vision_model}")
    print(f"   Answering Model: {args.answering_model}")
    print(f"   4-bit Quantization: {args.use_4bit}")
    print()
    
    try:
        pipeline = RAGVQAPipeline(
            vision_model_name=args.vision_model,
            answering_model_name=args.answering_model,
            use_4bit=args.use_4bit,
            kb_path="../data/knowledge_base.json",
            vector_db_path="../models/vector_db"
        )
        print(" Pipeline initialized successfully!")
        print()
    except Exception as e:
        print(f" Lỗi khi khởi tạo pipeline: {e}")
        print()
        print("Gợi ý:")
        print("  - Kiểm tra đã cài đặt dependencies: pip install -r code/requirements.txt")
        print("  - Kiểm tra có GPU và đủ VRAM (khuyến nghị 16GB+)")
        print("  - Kiểm tra file data/knowledge_base.json có tồn tại")
        return
    
    # Load image
    print(" Loading image...")
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"   Image size: {image.size}")
        print()
    except Exception as e:
        print(f"    Lỗi khi load hình ảnh: {e}")
        return
    
    # Process
    print(" Processing...")
    print("-" * 80)
    
    try:
        result = pipeline.process(
            image=image,
            question=args.question,
            return_intermediate=True
        )
        
        # Print results
        print()
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print()
        
        print("1️.  CAPTION (Mô tả hình ảnh):")
        print("-" * 80)
        print(result.get("caption", "N/A"))
        print()
        
        print("2️. OCR (Văn bản trong ảnh):")
        print("-" * 80)
        ocr = result.get("ocr", "")
        print(ocr if ocr else "(Không có văn bản)")
        print()
        
        print("3️.  RETRIEVED DOCUMENTS (Tài liệu liên quan):")
        print("-" * 80)
        docs = result.get("retrieved_docs", [])
        if docs:
            for i, doc in enumerate(docs, 1):
                print(f"\n[{i}] {doc[:200]}..." if len(doc) > 200 else f"\n[{i}] {doc}")
        else:
            print("(Không tìm thấy tài liệu liên quan)")
        print()
        
        print("4️.  FINAL ANSWER:")
        print("-" * 80)
        print(result.get("answer", "N/A"))
        print()
        
        print("=" * 80)
        print(" Demo completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n Lỗi khi xử lý: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("Gợi ý:")
        print("  - Nếu lỗi Out of Memory: giảm batch size hoặc dùng GPU lớn hơn")
        print("  - Nếu lỗi model: kiểm tra kết nối internet để download từ HuggingFace")


if __name__ == "__main__":
    main()
