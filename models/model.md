# Fine-tuned Model: qwen2vl-7b-vqa-grounded

## Download Link
Google Drive: https://drive.google.com/file/d/1E_5xOVOJSL_zZtEd5SrIo_5IeCZNqx2M/view?usp=sharing

## Thông tin Model
- Tên: qwen2vl-7b-vqa-grounded
- Kích thước: ~5GB (nén)
- Base model: Qwen2-VL-7B-Instruct
- Fine-tuning method: QLoRA
- Dataset: Vietnamese VQA for Cultural Heritage

## Hướng dẫn tải và cài đặt

### Cách 1: Tải thủ công
1. Mở link Google Drive ở trên
2. Download file `qwen2vl-7b-vqa-grounded.zip`
3. Giải nén vào thư mục models/:
   ```bash
   unzip qwen2vl-7b-vqa-grounded.zip -d models/
   ```

### Cách 2: Dùng gdown
```bash
pip install gdown

# Thay FILE_ID bằng ID từ link Drive
gdown https://drive.google.com/uc?id=FILE_ID -O models/qwen2vl-7b-vqa-grounded.zip
unzip models/qwen2vl-7b-vqa-grounded.zip -d models/
```

### Cách 3: Dùng gdown với link đầy đủ
```bash
pip install gdown
gdown "LINK_GOOGLE_DRIVE_ĐẦY_ĐỦ" -O models/qwen2vl-7b-vqa-grounded.zip
unzip models/qwen2vl-7b-vqa-grounded.zip -d models/
```

## Kiểm tra sau khi cài đặt
```bash
# Kiểm tra cấu trúc
ls models/qwen2vl-7b-vqa-grounded/

# Kết quả mong đợi:
# config.json
# model-00001-of-00003.safetensors
# model-00002-of-00003.safetensors
# model-00003-of-00003.safetensors
# tokenizer_config.json
# tokenizer.json
# ...
```

## Sử dụng model
```bash
# Chạy demo với fine-tuned model
python code/run_demo.py --answering-model models/qwen2vl-7b-vqa-grounded

# Hoặc để script tự động detect
python code/run_demo.py
```
