"""
Centralized prompts for the RAG-enhanced VQA system.
Ensures consistency between training (fine-tuning) and inference.

IMPORTANT: Any changes here should be reflected in:
1. code/src/answering.py (inference)
2. code/scripts/3_convert_to_sharegpt.py (fine-tuning data)
"""

# System prompt - Dùng cho cả inference và fine-tuning
SYSTEM_PROMPT = (
    "Bạn là trợ lý VQA chuyên về lịch sử và văn hóa Việt Nam. "
    "Trả lời câu hỏi dựa trên HÌNH ẢNH là chính. "
    "Mô tả hình ảnh và văn bản OCR chỉ là thông tin tham khảo, có thể không chính xác. "
    "Kiến thức bổ sung giúp giải thích sâu hơn khi cần.\n\n"
    "QUY TẮC BẮT BUỘC:\n"
    "1. CHỈ trả lời bằng tiếng Việt thuần túy, TUYỆT ĐỐI KHÔNG dùng tiếng Trung, Hàn, Nhật hay bất kỳ ngôn ngữ nào khác.\n"
    "2. Sử dụng chính tả tiếng Việt chuẩn với dấu thanh đúng.\n"
    "3. Trả lời rõ ràng, có cấu trúc và dễ hiểu."
)


def build_user_prompt(
    caption: str,
    ocr_text: str,
    context: str,
    question: str
) -> str:
    """
    Build user prompt for VQA task.
    
    Args:
        caption: Image caption (from vision module)
        ocr_text: OCR text extracted from image
        context: Retrieved knowledge context (from RAG)
        question: User question
        
    Returns:
        Formatted user prompt string
    """
    return (
        f"Dựa vào mô tả hình ảnh: {caption}\n\n"
        f"Thông tin văn bản trong ảnh: {ocr_text}\n\n"
        f"Kiến thức lịch sử và văn hóa:\n{context}\n\n"
        f"Câu hỏi: {question}\n\n"
        "Hãy trả lời bằng tiếng Việt chuẩn (KHÔNG dùng tiếng Trung hay ngôn ngữ khác). "
        "Nếu phù hợp, hãy cung cấp giải thích chi tiết, ý nghĩa văn hóa và bối cảnh lịch sử."
    )


# Default no-context message
NO_CONTEXT_MESSAGE = "Không có thông tin liên quan."

# Default OCR message when empty
NO_OCR_MESSAGE = "Không có văn bản."

# Default caption message when empty
NO_CAPTION_MESSAGE = "Không có mô tả hình ảnh."
