# Audio Dataset Pipeline

Pipeline giúp đóng gói dữ liệu audio và transcript thành định dạng Hugging Face Dataset để finetune STT/TTS.

## Cài đặt

```bash
pip install datasets torchcodec ffmpeg-python
```
Yêu cầu: `ffmpeg` đã được cài đặt trên hệ thống.

## Hướng dẫn sử dụng (CLI)

### 1. Chuẩn bị file Config (JSON)

Tạo file `config.json` định nghĩa các file audio cần xử lý:

```json
{
  "path/to/hatinh1.mp3": {
    "segments": [
      ["00:38-00:41"], 
      ["00:43-01:11"]
    ],
    "transcript_path": "path/to/transcript_hatinh1.txt"
  },
  "path/to/audio2.mp3": {
    "segments": [["01:00-01:10"]],
    "transcript_path": "path/to/transcript_audio2.txt"
  }
}
```
**Lưu ý:** File transcript phải chứa câu marker (ví dụ: "Đây là đoạn âm thanh dùng để phân tách") để phân biệt các đoạn.

### 2. Chạy lệnh tạo Dataset

```bash
python run_pipeline.py --input-json config.json --output-dir my_dataset --dataset-name vn_voice
```

### 3. Kết quả (Output)

Trong thư mục `my_dataset/`:
- `wavs/`: Các file audio đã cắt nhỏ (Format: mono, 16kHz).
- `metadata.csv`: File chứa thông tin mapping (ID, path, text, duration) để kiểm tra.
- `hf_dataset/`: Dữ liệu dạng binary Hugging Face, có thể load bằng python:

```python
from datasets import load_from_disk
dataset = load_from_disk("my_dataset/hf_dataset")
print(dataset[0])
```

---

## (Tools) Các tính năng khác

- **`merge_segments_with_marker`** (trong `edit_audio.py`): Gộp audio và chèn marker để gửi cho Gemini transcribe.
- **`map_transcript_to_segments`**: Map text trả về từ Gemini vào từng segment gốc.

## (Legacy) Diarization cũ

Tham khảo file `infer.py` để chạy flow cũ (Gemini API Call trực tiếp cho từng đoạn nhỏ).
