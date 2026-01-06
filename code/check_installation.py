#!/usr/bin/env python3
"""
Script kiểm tra installation và dependencies
Chạy trước khi run demo/evaluation

Usage:
    python check_installation.py
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print(" Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("     Warning: Python 3.10+ recommended")
        return False
    else:
        print("    Python version OK")
        return True


def check_dependencies():
    """Check if required packages are installed"""
    print("\n Checking dependencies...")
    
    required = [
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "sentence_transformers",
        "faiss",
        "PIL",
        "numpy",
        "tqdm"
    ]
    
    missing = []
    for package in required:
        try:
            if package == "PIL":
                __import__("PIL")
            elif package == "faiss":
                try:
                    __import__("faiss")
                except ImportError:
                    __import__("faiss_cpu")
            else:
                __import__(package)
            print(f"   {package}")
        except ImportError:
            print(f"   {package} - NOT FOUND")
            missing.append(package)
    
    if missing:
        print(f"\n   Missing packages: {', '.join(missing)}")
        print(f"   Run: pip install -r code/requirements.txt")
        return False
    else:
        print("   All dependencies OK")
        return True


def check_gpu():
    """Check GPU availability"""
    print("\n Checking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"    GPU: {gpu_name}")
            print(f"    VRAM: {gpu_memory:.1f} GB")
            
            if gpu_memory < 8:
                print(f"     Warning: Low VRAM. Recommend 16GB+")
            return True
        else:
            print("   Warning: No GPU detected. Will run on CPU (very slow)")
            return False
    except Exception as e:
        print(f"   Error checking GPU: {e}")
        return False


def check_data_files():
    """Check if data files exist"""
    print("\n Checking data files...")
    
    files = [
        ("../data/knowledge_base.json", "Knowledge Base"),
        ("../data/data-benchmark/benchmark_60.json", "Benchmark dataset"),
        ("../data/data-benchmark/benchmark_images", "Benchmark images"),
        ("../models/vector_db", "Vector DB index (optional)")
    ]
    
    all_ok = True
    for filepath, name in files:
        path = Path(filepath)
        if path.exists():
            if path.is_file():
                size = path.stat().st_size / 1024**2  # MB
                print(f"    {name}: {size:.1f} MB")
            else:
                print(f"    {name}: found")
        else:
            if "optional" in name.lower():
                print(f"    {name}: not found (will build if needed)")
            else:
                print(f"    {name}: NOT FOUND")
                all_ok = False
    
    return all_ok


def check_code_structure():
    """Check if code structure is correct"""
    print("\n Checking code structure...")
    
    modules = [
        "src/config.py",
        "src/pipeline.py",
        "src/vision.py",
        "src/retrieval.py",
        "src/answering.py",
        "src/prompts.py"
    ]
    
    all_ok = True
    for module in modules:
        path = Path(module)
        if path.exists():
            print(f"    {module}")
        else:
            print(f"    {module}: NOT FOUND")
            all_ok = False
    
    return all_ok


def main():
    print("=" * 80)
    print("RAG-VQA INSTALLATION CHECK")
    print("=" * 80)
    print()
    
    results = {
        "Python version": check_python_version(),
        "Dependencies": check_dependencies(),
        "GPU": check_gpu(),
        "Data files": check_data_files(),
        "Code structure": check_code_structure()
    }
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for check, status in results.items():
        print(f"{check}: {'OK' if status else 'FAIL'}")
    
    print()
    
    if all(results.values()):
        print(" All checks passed! You can now run:")
        print("   python run_demo.py")
        print("   python run_evaluation.py")
    else:
        print("  Some checks failed. Please fix the issues above.")
        print()
        print("Common fixes:")
        print("  - Install dependencies: pip install -r code/requirements.txt")
        print("  - Check data files are in correct locations")
        print("  - Ensure you have GPU with CUDA installed")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
