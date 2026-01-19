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
python infer.py --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" --output ./outputs/output.rttm --segment-seconds 120 --model gemini-3-flash-preview
```

Audio file local:

```
python infer.py --audio-file /path/to/audio.mp3 --file-id my_audio --output ./outputs/output.rttm --segment-seconds 120 --model gemini-3-flash-preview
```

Tham số quan trọng:

- `--segment-seconds`: thời lượng mỗi đoạn, mặc định 600
- `--time-format`: định dạng timestamp, `hms` (mặc định) hoặc `seconds`
- `--model`: model Vertex AI, mặc định gemini-3-flash-preview
- `--output-dir`: thư mục lưu audio tải về
- `--api-key`: dùng khi không set env

## Token và chi phí ước tính

Sau khi chạy sẽ in ra usage ở stderr:

```
usage input_tokens=1200 output_tokens=3400 total_tokens=4600 estimated_cost_usd=0.010200 pricing=gemini-3-flash-preview
```

Ước tính token dùng công thức `ceil(len(text) / 4)` cho prompt và output text, không tính audio tokens.

Bảng giá tham chiếu:

- gemini-3-flash-preview: $0.50 input / $3 output (per 1M tokens)
- gemini-3-pro-preview: $2 / $12 (<200k tokens), $4 / $18 (>=200k tokens)

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

Role: Bạn là chuyên gia Audio Processing và Speech-to-Text.
Task: Thực hiện Speaker Diarization và Transcription đồng thời cho đoạn audio hiện tại.

Ràng buộc đầu ra:

- Chỉ trả về kết quả, không giải thích, không JSON, không markdown.
- Mỗi dòng theo format: `<file_id> <speaker_name_or_role> <start_time> <end_time> <transcript> <gender>`
- file_id phải đúng.
- start_time/end_time là số giây (float) trong đoạn audio hiện tại (0-based).
- Không bọc transcript trong dấu ngoặc kép.
- Nếu không xác định giới tính, ghi `unknown`.

Quy tắc nội dung:

- Bỏ qua khoảng lặng, chỉ ghi đoạn có tiếng nói.
- Bỏ qua quảng cáo, hát, nhạc nền; chỉ tập trung phần hội thoại cuộc họp.
- Không tách theo câu; gộp liên tục theo người nói cho đến khi người khác nói.
- Giữ tên/chức vụ speaker nhất quán trong đoạn.

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

## edit audio file
```
python edit_audio.py --input data/audio.mp3 --output-dir outputs/ --segments "0:00-1:30,1:30-3:00" --name "kyhopl11"
```
