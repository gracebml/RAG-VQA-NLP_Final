#!/usr/bin/env python3
"""
Script chạy evaluation trên benchmark dataset
Sử dụng cho LOCAL testing

Usage:
    python run_evaluation.py
    python run_evaluation.py --num-samples 10  # Test với 10 samples
    python run_evaluation.py --model path/to/finetuned/model
"""

import sys
import argparse
from pathlib import Path
import json
from tqdm import tqdm

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import RAGVQAPipeline
from PIL import Image
from datetime import datetime
import os


def get_default_answering_model():
    """Tìm fine-tuned model, fallback về base model nếu không có"""
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
    
    print("     Fine-tuned model not found, using base model")
    return "Qwen/Qwen2-VL-2B-Instruct"


def load_benchmark_data(benchmark_path, num_samples=None):
    """Load benchmark dataset"""
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if num_samples:
        data = data[:num_samples]
    
    return data


def calculate_metrics(predictions, ground_truths):
    """Calculate simple metrics (without heavy dependencies)"""
    # Simple metrics - for full metrics use notebook
    results = {
        "total_samples": len(predictions),
        "avg_answer_length": sum(len(p) for p in predictions) / len(predictions),
        "samples_with_answer": sum(1 for p in predictions if p and len(p) > 0)
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG-VQA on Benchmark")
    parser.add_argument(
        "--benchmark", 
        type=str, 
        default="../data/data-benchmark/benchmark_60.json",
        help="Đường dẫn đến benchmark file"
    )
    parser.add_argument(
        "--image-dir", 
        type=str, 
        default="../data/data-benchmark/benchmark_images",
        help="Thư mục chứa hình ảnh"
    )
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=None,
        help="Số lượng samples để test (None = tất cả)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Answering model path (mặc định: tự động tìm fine-tuned model)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results",
        help="Thư mục lưu kết quả"
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        default=True,
        help="Sử dụng 4-bit quantization"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RAG-VQA EVALUATION - Benchmark Testing")
    print("=" * 80)
    print()
    
    # Auto-detect fine-tuned model if not specified
    if args.model is None:
        args.model = get_default_answering_model()
    
    # Check files
    benchmark_path = Path(args.benchmark)
    image_dir = Path(args.image_dir)
    
    if not benchmark_path.exists():
        print(f" Không tìm thấy benchmark file: {benchmark_path}")
        return
    
    if not image_dir.exists():
        print(f" Không tìm thấy thư mục hình ảnh: {image_dir}")
        return
    
    # Load data
    print(f" Loading benchmark data from {benchmark_path}...")
    benchmark_data = load_benchmark_data(benchmark_path, args.num_samples)
    print(f"   Loaded {len(benchmark_data)} samples")
    print()
    
    # Initialize pipeline
    print(" Initializing RAG-VQA Pipeline...")
    print(f"   Model: {args.model}")
    print(f"   4-bit Quantization: {args.use_4bit}")
    print()
    
    try:
        pipeline = RAGVQAPipeline(
            vision_model_name="Qwen/Qwen2-VL-7B-Instruct",
            answering_model_name=args.model,
            use_4bit=args.use_4bit,
            kb_path="../data/knowledge_base.json",
            vector_db_path="../models/vector_db"
        )
        print(" Pipeline initialized!")
        print()
    except Exception as e:
        print(f" Lỗi khi khởi tạo pipeline: {e}")
        return
    
    # Run evaluation
    print(" Running evaluation...")
    print("-" * 80)
    
    results = []
    predictions = []
    ground_truths = []
    
    for idx, sample in enumerate(tqdm(benchmark_data, desc="Processing")):
        try:
            # Load image
            image_filename = sample.get("image_id", f"{idx:06d}.jpg")
            if not image_filename.endswith(('.jpg', '.jpeg', '.png')):
                image_filename = f"{image_filename}.jpg"
            
            image_path = image_dir / image_filename
            
            if not image_path.exists():
                print(f"\n  Warning: Image not found: {image_path}")
                continue
            
            image = Image.open(image_path).convert("RGB")
            question = sample["question"]
            ground_truth = sample.get("answer", "")
            
            # Process
            result = pipeline.process(
                image=image,
                question=question,
                return_intermediate=False
            )
            
            answer = result["answer"]
            
            # Save result
            results.append({
                "image_id": sample.get("image_id", f"{idx:06d}"),
                "question": question,
                "ground_truth": ground_truth,
                "prediction": answer,
                "caption": result.get("caption", ""),
                "ocr": result.get("ocr", "")
            })
            
            predictions.append(answer)
            ground_truths.append(ground_truth)
            
        except Exception as e:
            print(f"\n Error processing sample {idx}: {e}")
            continue
    
    print()
    print("=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)
    print()
    
    # Calculate metrics
    print(" Metrics:")
    metrics = calculate_metrics(predictions, ground_truths)
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    print()
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(args.model).name if "/" not in args.model else args.model.split("/")[-1]
    
    output_file = output_dir / f"evaluation_{model_name}_{timestamp}.json"
    
    output_data = {
        "config": {
            "model": args.model,
            "num_samples": len(results),
            "timestamp": timestamp
        },
        "metrics": metrics,
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f" Results saved to: {output_file}")
    print()
    
    # Print sample results
    print(" Sample Results (first 3):")
    print("-" * 80)
    for i, r in enumerate(results[:3], 1):
        print(f"\n[Sample {i}]")
        print(f"Question: {r['question']}")
        print(f"Ground Truth: {r['ground_truth']}")
        print(f"Prediction: {r['prediction']}")
        print("-" * 80)
    
    print()
    print(" Evaluation completed successfully!")
    print()
    print(" Tips:")
    print("   - Để tính toán metrics đầy đủ (BERTScore, ROUGE-L), sử dụng notebook 4-evaluation.ipynb")
    print("   - Kết quả đã lưu dưới dạng JSON để xử lý thêm")
    print("=" * 80)


if __name__ == "__main__":
    main()
