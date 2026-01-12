# Gemini_diarization_audio

Mục tiêu: nhận transcript từ Youtube video hoặc file audio bằng Gemini Vertex AI API và LangChain.

Input: youtube video url hoặc audio file
Output: speaker, timestamp, transcript

## Flow

1. Tải audio từ YouTube hoặc dùng file local
2. Tách audio theo đoạn cố định để tối ưu kích thước request
3. Gọi Gemini Vertex AI API cho từng đoạn
4. Chuẩn hóa và cộng offset thời gian cho kết quả

## Yêu cầu

- Python 3.10+
- ffmpeg, ffprobe
- API key cho Gemini Vertex AI

## Cài đặt

```
pip install -r requirements.txt
```

Thiết lập API key:

```
export GEMINI_API_KEY=YOUR_KEY
```

## Chạy diarization + transcription

YouTube URL:

```
python infer.py --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" --output output.rttm
```

Audio file local:

```
python infer.py --audio-file /path/to/audio.mp3 --file-id my_audio --output output.rttm
```

Tham số quan trọng:

- `--segment-seconds`: thời lượng mỗi đoạn, mặc định 600
- `--model`: model Vertex AI, mặc định gemini-3-flash-preview
- `--output-dir`: thư mục lưu audio tải về
- `--api-key`: dùng khi không set env

## Format output

Nghiêm ngặt - RTTM Hybrid:

```
<file_id> <name or position of who representation> <start_time> <end_time> <transcript> <gender>
```

Ví dụ:

```
video123 SPEAKER_01 0.52 3.10 Xin chào unknown
video123 SPEAKER_02 3.55 6.20 Tôi nghe đây unknown
```

## Prompt sử dụng

Role: Bạn là một chuyên gia về Audio Processing và Speech-to-Text.
Task: Phân tích file audio, thực hiện Speaker Diarization và Transcription đồng thời.

Yêu cầu kỹ thuật:

- Nhận diện chính xác thời điểm bắt đầu (start_time) và kết thúc (end_time) của mỗi phân đoạn.
- Gán Speaker ID nhất quán (SPEAKER_01, SPEAKER_02, ...).
- Nếu có khoảng trống thì bỏ qua, bắt đầu tính start_time từ lúc có giọng nói.

Format Output (Nghiêm ngặt - RTTM Hybrid):

```
<file_id> <name or position of who representation> <start_time> <end_time> <transcript> <gender>
```

## API call mẫu

```
curl --location 'https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?key=MY_KEY' \
--header 'Content-Type: application/json' \
--data '{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "Explain how AI works in a few words"
        }
      ]
    }
  ]
}'
```
