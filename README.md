# RAG-Enhanced Visual Question Answering cho Lịch sử & Văn hóa Việt Nam

Hệ thống VQA (Visual Question Answering) được tăng cường bằng RAG (Retrieval Augmented Generation) để trả lời câu hỏi về lịch sử và văn hóa Việt Nam dựa trên hình ảnh.

## Quick Start

```bash
# 1. Cài đặt
pip install -r code/requirements.txt

# 2. Kiểm tra (tùy chọn)
python code/check_installation.py

# 3. Chạy demo
python code/run_demo.py

# 4. Evaluation (tùy chọn)
python code/run_evaluation.py --num-samples 10
```

**Lưu ý:** Để chạy trên Kaggle, xem [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md)

---

## Tổng quan Đồ án

### Mục tiêu
Xây dựng hệ thống trả lời câu hỏi về lịch sử và văn hóa Việt Nam dựa trên hình ảnh, kết hợp:
- Vision Language Model (Qwen2-VL) để hiểu hình ảnh
- RAG (Retrieval Augmented Generation) để tìm kiếm thông tin từ knowledge base
- Fine-tuning model để cải thiện chất lượng câu trả lời

### Kiến trúc Hệ thống

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG-VQA Pipeline                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────┐    ┌───────────────┐    ┌───────────────┐    ┌─────────┐  │
│  │  IMAGE  │───▶│ Vision Module │───▶│   Retrieval   │───▶│Answering│  │
│  │    +    │    │ (Qwen2-VL-7B) │    │    Module     │    │ Module  │  │
│  │QUESTION │    │               │    │               │    │         │  │
│  └─────────┘    │ • Caption     │    │ • BM25        │    │Qwen2-VL │  │
│                 │ • OCR         │    │ • Embedding   │    │  2B     │  │
│                 └───────────────┘    │ • Wikipedia   │    │(tuned)  │  │
│                                      └───────────────┘    └─────────┘  │
│                                                                         │
│                              ▼                                          │
│                     ┌───────────────┐                                   │
│                     │   ANSWER      │                                   │
│                     │               |                                    │
│                     └───────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Các thành phần chính

1. **Vision Module**: Xử lý hình ảnh
   - Caption: Mô tả nội dung hình ảnh bằng tiếng Việt
   - OCR: Đọc văn bản trong hình ảnh (bia đá, câu đối, v.v.)

2. **Retrieval Module**: Tìm kiếm thông tin
   - BM25: Tìm kiếm dựa trên từ khóa
   - Embedding: Tìm kiếm semantic
   - Wikipedia: Bổ sung thông tin từ Wikipedia tiếng Việt

3. **Answering Module**: Tạo câu trả lời
   - Sử dụng Qwen2-VL-2B đã fine-tune
   - Kết hợp thông tin từ hình ảnh và knowledge base

## Cấu trúc thư mục

```
NLP-Final-Prj/
├── README.md                   # File hướng dẫn này
│
├── code/                       # Mã nguồn
│   ├── run_demo.py             # Script chạy demo đơn giản (LOCAL)
│   ├── run_evaluation.py       # Script chạy evaluation (LOCAL)
│   ├── check_installation.py   # Script kiểm tra cài đặt (LOCAL)
│
├── data/                       # Dữ liệu
│   ├── knowledge_base.json     # Knowledge Base (~45K entries)
│   ├── vqa_train.json          # Training data
│   ├── vqa_test.json           # Test data
│   └── data-benchmark/         # Benchmark dataset (60 samples)
│       ├── benchmark_60.json
│       └── benchmark_images/   # 60 hình ảnh test
│
├── models/                     # Models và index
│   ├── model.txt              # Link Google Drive để tải fine-tuned model
│   └── vector_db/              # FAISS index (pre-built)
│       ├── vector_db.index
│       └── vector_db_config.json
│
├── results/                    # Kết quả evaluation
│   ├── benchmark_inference_results_*.json
│   └── model_comparison_*.json
│
└── code/                       # Source code
    ├── requirements.txt        # Dependencies
    │
    ├── src/                    # Core modules
    │   ├── config.py           # Cấu hình (tự động detect Kaggle/Local)
    │   ├── pipeline.py         # Main RAG-VQA Pipeline
    │   ├── vision.py           # Vision Module
    │   ├── retrieval.py        # Retrieval Module
    │   ├── answering.py        # Answering Module
    │   └── prompts.py          # Prompts template
    │
    ├── scripts/                # Data preprocessing scripts
    │   ├── 0_preprocess_vqa.py
    │   ├── 1_merge_caption_ocr.py
    │   ├── 2_split_train_test.py
    │   └── 3_convert_to_sharegpt.py
    │
    ├── notebooks/              # Jupyter notebooks (KAGGLE)
    │   ├── 1_Build_Index.ipynb              # Xây dựng vector index
    │   ├── 2-inference-sample.ipynb         # Demo inference
    │   ├── 3-finetune-qwen2vl-llamafactory.ipynb  # Fine-tuning
    │   └── 4-evaluation.ipynb               # Evaluation đầy đủ
    │
    └── finetuning/             # Fine-tuning configs
        ├── llamafactory_config.yaml
        ├── dataset_info.json
        └── export_model.yaml
```

## Yêu cầu Hệ thống

### Chạy Local
- Python 3.10+
- GPU: NVIDIA GPU với ít nhất 16GB VRAM (khuyến nghị 24GB)
  - Hoặc dùng 4-bit quantization cho GPU 8GB+
- RAM: 16GB+
- Storage: 50GB+ (để tải models)

### Chạy trên Kaggle
- Kaggle Notebook với GPU T4/P100
- Internet để tải models từ HuggingFace

## Hướng dẫn Cài đặt và Chạy

### A. Chạy trên LOCAL (Khuyến nghị cho kiểm tra nhanh)

#### Bước 1: Cài đặt Dependencies

```bash
# Cài đặt các thư viện cần thiết
pip install -r code/requirements.txt
```

#### Bước 2: Chuẩn bị Dữ liệu và Model

**Dữ liệu** đã có sẵn trong thư mục `data/`:
- `data/knowledge_base.json`: Knowledge base (~45K entries)
- `data/data-benchmark/benchmark_60.json`: 60 câu hỏi test
- `data/data-benchmark/benchmark_images/`: Hình ảnh tương ứng
- `models/vector_db/`: Vector index (đã build sẵn)

**Fine-tuned Model** (Khuyến nghị):

Model đã được fine-tune (qwen2vl-7b-vqa-grounded) được lưu trữ trên Google Drive do kích thước lớn (~6GB).

**Cách tải model:** **Xem hướng dẫn chi tiết:** [models/model.md](models/model.md)

#### Bước 3: Chạy Demo Đơn Giản

**Cách 1: Sử dụng script có sẵn**

```bash
# Từ thư mục gốc NLP-Final-Prj/
python code/run_demo.py
```

Script này sẽ:
1. Tự động tìm fine-tuned model (hoặc dùng base model nếu không có)
2. Chạy inference trên 1 ví dụ mẫu
3. In ra caption, OCR, retrieved docs, và answer

**Lưu ý:** Để dùng fine-tuned model, đặt model vào một trong các thư mục:
- `models/finetuned/qwen2vl-2b-vqa`
- `models/qwen2vl-2b-vqa-finetuned`
- `models/finetuned`

Hoặc chỉ định model trực tiếp:
```bash
python code/run_demo.py --answering-model path/to/your/finetuned/model
```

**Cách 2: Sử dụng Python code trực tiếp**

```python
# File: test_pipeline.py
import sys
sys.path.insert(0, 'code')

from src.pipeline import RAGVQAPipeline
from PIL import Image

# Initialize pipeline
pipeline = RAGVQAPipeline(
    vision_model_name="Qwen/Qwen2-VL-7B-Instruct",
    answering_model_name="../models/qwen2vl-7b-vqa-grounded",  # Fine-tuned model
    # answering_model_name="Qwen/Qwen2-VL-2B-Instruct",  # Hoặc dùng base model
    use_4bit=True,  # Sử dụng 4-bit quantization
    kb_path="../data/knowledge_base.json",
    vector_db_path="../models/vector_db"
)

# Load image
image = Image.open("data/data-benchmark/benchmark_images/000000.jpg")

# Ask question
question = "Đây là công trình kiến trúc gì? Được xây dựng khi nào?"

# Get answer
result = pipeline.process(image, question, return_intermediate=True)

# Print results
print("=" * 80)
print("CAPTION:", result["caption"])
print("=" * 80)
print("OCR:", result["ocr"])
print("=" * 80)
print("RETRIEVED DOCS:")
for i, doc in enumerate(result["retrieved_docs"], 1):
    print(f"{i}. {doc}")
print("=" * 80)
print("ANSWER:", result["answer"])
print("=" * 80)
```

#### Bước 4: Chạy Evaluation trên Benchmark

```bash
# Test với 10 samples (nhanh, ~20 phút)
python code/run_evaluation.py --num-samples 10

# Test toàn bộ 60 samples (~1-2 giờ)
python code/run_evaluation.py
```

Script này sẽ:
1. Load câu hỏi từ benchmark
2. Chạy inference với fine-tuned model + RAG
3. Tính toán metrics
4. Lưu kết quả vào `results/`

### B. Chạy trên KAGGLE (Cho Fine-tuning và Evaluation đầy đủ)

**Xem hướng dẫn chi tiết:** [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md)

Tóm tắt:
1. Upload datasets lên Kaggle
2. Tạo notebook với GPU T4/P100
3. Chạy notebooks theo thứ tự
4. Download results về

#### Các notebooks quan trọng:

- **1_Build_Index.ipynb**: Xây dựng vector index (~30 phút)
- **2-inference-sample.ipynb**: Demo inference (~10 phút)
- **3-finetune-qwen2vl-llamafactory.ipynb**: Fine-tuning (~3 giờ)
- **4-evaluation.ipynb**: Evaluation đầy đủ (~1 giờ)

---

## Cấu hình Chi tiết

### File config.py

File `code/src/config.py` tự động detect môi trường (Kaggle/Local):

```python
# Tự động detect Kaggle
IS_KAGGLE = Path('/kaggle/working').exists()

# Paths tự động điều chỉnh
if IS_KAGGLE:
    BASE_DIR = Path('/kaggle/working')
    DATA_DIR = Path('/kaggle/input')
else:
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
```

### Các tham số quan trọng

```python
# Models
QWEN2VL_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"  # Vision model
VIETNAMESE_EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"  # Embedding

# Retrieval
RETRIEVAL_METHOD = "hybrid"  # "bm25" | "embedding" | "hybrid"
TOP_K_RETRIEVE = 3           # Số documents trả về
WIKIPEDIA_FALLBACK = True    # Có dùng Wikipedia không

# Generation
MAX_NEW_TOKENS = 256         # Độ dài tối đa câu trả lời
TEMPERATURE = 0.7            # Nhiệt độ sampling (0-1)
```

## Các Module Chính

### 1. Vision Module (`src/vision.py`)

```python
from src.vision import VisionModule

vision = VisionModule(model_name="Qwen/Qwen2-VL-2B-Instruct")

# Generate caption
caption = vision.generate_caption(image)

# Extract OCR
ocr_text = vision.extract_ocr(image)
```

### 2. Retrieval Module (`src/retrieval.py`)

```python
from src.retrieval import RetrievalModule

retrieval = RetrievalModule(
    kb_path="data/knowledge_base.json",
    vector_db_path="models/vector_db"
)

# Retrieve documents
docs = retrieval.retrieve(
    query="Chùa Một Cột",
    caption="Một ngôi chùa cổ ở Hà Nội",
    ocr_text="",
    top_k=3
)
```

### 3. Answering Module (`src/answering.py`)

```python
from src.answering import AnsweringModule

answering = AnsweringModule(model_name="Qwen/Qwen2-VL-7B-Instruct")

# Generate answer
answer = answering.generate_answer(
    image=image,
    question="Đây là gì?",
    caption=caption,
    ocr_text=ocr_text,
    context=retrieved_docs
)
```

### 4. Complete Pipeline (`src/pipeline.py`)

```python
from src.pipeline import RAGVQAPipeline

pipeline = RAGVQAPipeline(
    vision_model_name="Qwen/Qwen2-VL-7B-Instruct",
    answering_model_name="models/finetuned/qwen2vl-2b-vqa",  # Model đã fine-tune
    # answering_model_name="Qwen/Qwen2-VL-2B-Instruct",  # Hoặc base model
    use_4bit=True
)

result = pipeline.process(image, question, return_intermediate=True)
```

## Evaluation Metrics

| Metric | Mô tả | Công thức |
|--------|-------|-----------|
| BERTScore F1 | Đo semantic similarity với ground truth | F1 score của BERT embeddings |
| ROUGE-L F1 | Đo overlap của longest common subsequence | F1 của LCS |
| Faithfulness | Đo độ trung thực với RAG context | NLI score giữa answer và context |

## Kết quả Benchmark

Kết quả trên 60 samples (có sẵn trong `results/`):

| Phương pháp | BERTScore | ROUGE-L | Faithfulness |
|-------------|-----------|---------|--------------|
| Baseline 1 (Zero-shot) | 0.713 | 0.316 | N/A |
| Baseline 2 (RAG + Base) | 0.718 | 0.308 | 0.027 |
| **Proposed (RAG + Fine-tuned)** | **0.738** | **0.357** | **0.029** |

## Troubleshooting

### Lỗi Out of Memory (OOM)

```python
# Đảm bảo dùng 4-bit quantization
pipeline = RAGVQAPipeline(use_4bit=True)

# Giảm batch size hoặc max tokens
MAX_NEW_TOKENS = 128  # trong config.py
```

### Lỗi không tìm thấy module

```bash
# Đảm bảo chạy từ đúng thư mục
cd NLP-Final-Prj
python code/run_demo.py

# Hoặc thêm PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/code"
```

### Lỗi download model

```python
# Model sẽ tự động download từ HuggingFace
# Đảm bảo có kết nối internet
# Hoặc download trước và chỉ đến local path:

vision_model_name = "/path/to/local/qwen2-vl-7b"
```

### Lỗi Kaggle kernel timeout

```python
# Giảm số samples test
# Trong notebook, thay vì chạy hết 60 samples:
benchmark_data = benchmark_data[:10]  # Test với 10 samples trước
```

## Tài liệu Tham khảo

- [Qwen2-VL Model](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Vietnamese SBERT](https://huggingface.co/keepitreal/vietnamese-sbert)
- [FAISS](https://github.com/facebookresearch/faiss)

## Liên hệ

NLP Final Project - Vietnamese Cultural VQA with RAG

Nếu có vấn đề khi chạy code, vui lòng kiểm tra:
1. Đã cài đặt đủ dependencies chưa (`pip install -r code/requirements.txt`)
2. Đã có đủ dữ liệu trong thư mục `data/` chưa
3. GPU có đủ VRAM không (khuyến nghị 16GB+)
4. File config.py có detect đúng môi trường không
