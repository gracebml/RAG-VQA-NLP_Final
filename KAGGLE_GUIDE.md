# Hướng dẫn Upload và Chạy trên Kaggle

## Bước 1: Chuẩn bị Datasets

### Dataset 1: Knowledge Base
1. Tạo dataset mới trên Kaggle: "knowledge-base-data"
2. Upload file: `data/knowledge_base.json`

### Dataset 2: Benchmark Data
1. Tạo dataset mới: "vqa-benchmark-data"
2. Upload:
   - `data/data-benchmark/benchmark_60.json`
   - Thư mục `data/data-benchmark/benchmark_images/` (60 hình ảnh)

### Dataset 3: Training Data (cho Fine-tuning)
1. Tạo dataset mới: "vqa-training-data"
2. Upload:
   - `code/finetuning/llama-dataset/vqa_vietnamese_train.json`
   - `code/finetuning/llama-dataset/vqa_vietnamese_test.json`

### Dataset 4: Vector DB (Optional)
1. Tạo dataset mới: "vector-db-index"
2. Upload thư mục: `models/vector_db/`
3. (Nếu không có, notebook sẽ build lại)

### Dataset 5: Fine-tuned Model
1. **Download model từ Google Drive:**
   - Xem link trong file `models/model.txt`
   - Model: `qwen2vl-7b-vqa-grounded` (~6GB)
   - Giải nén để có thư mục chứa `config.json`, `model.safetensors`, etc.

2. **Tạo dataset trên Kaggle:**
   - Tạo dataset mới: "qwen2vl-7b-vqa-grounded" hoặc "finetuned-qwen-vqa"
   - Upload toàn bộ thư mục model đã giải nén
   - Files cần có:
     ```
     qwen2vl-7b-vqa-grounded/
     ├── config.json
     ├── generation_config.json
     ├── model.safetensors (hoặc các file .bin)
     ├── tokenizer.json
     ├── tokenizer_config.json
     ├── preprocessor_config.json
     └── adapter_config.json (nếu có)
     ```

3. **Add dataset vào notebook:**
   - Settings -> Add Dataset -> Search "qwen2vl-7b-vqa-grounded"
   - Model sẽ có path: `/kaggle/input/qwen2vl-7b-vqa-grounded/`

**Lưu ý:** Nếu không có fine-tuned model, notebook sẽ tự động dùng base model từ HuggingFace.

## Bước 2: Tạo Kaggle Notebook

### 2.1. Tạo Notebook mới
1. Vào Kaggle -> Code -> New Notebook
2. Đặt tên: "RAG-VQA-Demo" (hoặc tên khác)

### 2.2. Cấu hình Notebook
1. Settings -> Accelerator -> **GPU T4 x2** (hoặc P100)
2. Settings -> Internet -> **ON** (để download models)
3. Add Datasets:
   - Add "knowledge-base-data"
   - Add "vqa-benchmark-data"
   - Add "qwen2vl-7b-vqa-grounded" (fine-tuned model - **quan trọng!**)
   - Add "vqa-training-data" (nếu fine-tune lại)
   - Add "vector-db-index" (nếu có)

### 2.3. Upload Source Code

**Cách 1: Upload thủ công**
- Upload từng file trong `code/src/` lên notebook
- Hoặc tạo dataset "ragvqa-source-code" chứa `code/src/`

**Cách 2: Copy-paste vào cells**
- Tạo cells mới và paste code từ từng file

## Bước 3: Chạy Notebooks

### Notebook A: Demo Nhanh (10 phút)

Sử dụng notebook: `code/notebooks/2-inference-sample.ipynb`

**Nội dung:**
1. Setup và install dependencies
2. Upload source code vào `/kaggle/working/code/src/`
3. Initialize pipeline
4. Chạy demo trên 2-3 samples
5. Hiển thị kết quả chi tiết

### Notebook B: Build Vector Index (20-30 phút)

Sử dụng notebook: `code/notebooks/1_Build_Index.ipynb`

**Nội dung:**
1. Load knowledge base
2. Generate embeddings với vietnamese-sbert
3. Build FAISS index
4. Save to `/kaggle/working/models/vector_db/`

**Kết quả:** Download thư mục `vector_db/` về để sử dụng local

### Notebook C: Fine-tuning (6-8 giờ)

Sử dụng notebook: `code/notebooks/3-finetune-qwen2vl-llamafactory.ipynb`

**Nội dung:**
1. Install LLaMA Factory
2. Prepare training data
3. Configure LoRA fine-tuning
4. Train model
5. Merge và export model

**Kết quả:** Download model về để sử dụng local

### Notebook D: Evaluation Đầy đủ (1-2 giờ)

Sử dụng notebook: `code/notebooks/4-evaluation.ipynb`

**Nội dung:**
1. Load 3 models: Base, Fine-tuned, Zero-shot
2. Run inference trên 60 benchmark samples
3. Calculate metrics: BERTScore, ROUGE-L, Faithfulness
4. Generate comparison charts
5. Export results to JSON/CSV

**Kết quả:** Download `results/` về

## Bước 4: Cấu trúc Paths trên Kaggle

Sau khi add datasets, paths sẽ như sau:

```
/kaggle/input/
├── knowledge-base-data/
│   └── knowledge_base.json
├── vqa-benchmark-data/
│   └── data-benchmark/
│       ├── benchmark_60.json
│       └── benchmark_images/
├── qwen2vl-7b-vqa-grounded/      # Fine-tuned model
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── vqa-training-data/
│   └── llama-dataset/
│       ├── vqa_vietnamese_train.json
│       └── vqa_vietnamese_test.json
└── vector-db-index/
    └── vector_db/
        ├── vector_db.index
        └── vector_db_config.json

/kaggle/working/
├── code/
│   └── src/
│       ├── config.py      # Auto-detect Kaggle
│       ├── pipeline.py
│       ├── vision.py
│       ├── retrieval.py
│       ├── answering.py
│       └── prompts.py
├── models/
│   └── vector_db/         # Built hoặc copied từ input
└── results/
    └── *.json             # Evaluation results
```

## Bước 5: Template Notebook Đơn Giản

Copy code này vào Kaggle notebook:

```python
# Cell 1: Install dependencies
!pip install -q transformers accelerate bitsandbytes sentence-transformers qwen-vl-utils faiss-cpu

# Cell 2: Setup paths
import sys
from pathlib import Path

# Upload source code vào working directory
!mkdir -p /kaggle/working/code/src

# Copy từ input nếu bạn đã tạo dataset source-code
# !cp -r /kaggle/input/ragvqa-source-code/src/* /kaggle/working/code/src/

# Hoặc paste code vào cells riêng (xem template bên dưới)

# Add to path
sys.path.insert(0, '/kaggle/working/code')

# Cell 3: Import modules
from src.pipeline import RAGVQAPipeline
from PIL import Image
import json

# Cell 4: Initialize pipeline với fine-tuned model
pipeline = RAGVQAPipeline(
    vision_model_name="Qwen/Qwen2-VL-7B-Instruct",
    answering_model_name="/kaggle/input/qwen2vl-7b-vqa-grounded",  # Fine-tuned model
    kb_path="/kaggle/input/knowledge-base-data/knowledge_base.json",
    vector_db_path="/kaggle/input/vector-db-index/vector_db",  # Nếu có
    use_4bit=True
)
    answering_model_name="Qwen/Qwen2-VL-7B-Instruct",
    use_4bit=True,
    kb_path="/kaggle/input/knowledge-base-data/knowledge_base.json",
    vector_db_path="/kaggle/input/vector-db-index/vector_db"
)

# Cell 5: Load sample
image_path = "/kaggle/input/vqa-benchmark-data/data-benchmark/benchmark_images/000000.jpg"
image = Image.open(image_path)
question = "Đây là công trình kiến trúc gì?"

# Cell 6: Process
result = pipeline.process(image, question, return_intermediate=True)

# Cell 7: Display results
print("CAPTION:", result["caption"])
print("\nOCR:", result["ocr"])
print("\nRETRIEVED DOCS:")
for doc in result["retrieved_docs"]:
    print("-", doc[:100])
print("\nANSWER:", result["answer"])
```

## Bước 6: Template cho từng Module (nếu paste thủ công)

Nếu không upload source code, paste vào cells:

### Cell: config.py
```python
%%writefile /kaggle/working/code/src/config.py
[Copy toàn bộ nội dung file code/src/config.py]
```

### Cell: vision.py
```python
%%writefile /kaggle/working/code/src/vision.py
[Copy toàn bộ nội dung file code/src/vision.py]
```

(Làm tương tự cho các file khác)

## Bước 7: Tips & Tricks

### Tiết kiệm thời gian
- Sử dụng dataset "vector-db-index" thay vì build lại
- Test với ít samples trước (10 thay vì 60)
- Sử dụng GPU T4 x2 để nhanh hơn

### Xử lý errors
- Nếu OOM: giảm batch size hoặc dùng 4-bit
- Nếu timeout: chia nhỏ thành nhiều notebooks
- Nếu model download chậm: chạy offline với cached model

### Download kết quả
```python
# Zip results
!zip -r results.zip /kaggle/working/results/
# Download file results.zip từ Output tab
```

## Bước 8: Checklist

Trước khi chạy notebook:
- [ ] Đã tạo đủ datasets
- [ ] Đã add datasets vào notebook
- [ ] Đã bật GPU
- [ ] Đã bật Internet
- [ ] Đã upload source code
- [ ] Đã kiểm tra paths

Sau khi chạy xong:
- [ ] Download results/
- [ ] Download model (nếu fine-tune)
- [ ] Download vector_db (nếu build)
- [ ] Save notebook

## Support

Nếu có lỗi, kiểm tra:
1. Paths có đúng không (Kaggle paths khác local)
2. File `config.py` có detect Kaggle không (`IS_KAGGLE = True`)
3. GPU có đủ memory không
4. Datasets đã add vào notebook chưa
