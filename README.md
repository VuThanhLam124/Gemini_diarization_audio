# Audio Dataset Pipeline

Repo hỗ trợ **2 chế độ** trong cùng `run_pipeline.py`:

1. **Audio-only (mặc định mới)**: bắt đầu từ thư mục audio, tự diarization bằng pyannote, map speaker từ CSV (nếu có), xuất `wavs + metadata.csv + hf_dataset`.
2. **Legacy config+transcript**: giữ nguyên luồng cũ dùng `config_*.json` + transcript marker.

## Cài đặt

### 1) Legacy (nhẹ)
```bash
pip install -r requirements.txt
```

### 2) Audio-only diarization (pyannote)
```bash
pip install -r requirements_diarization.txt
```
Yêu cầu hệ thống: `ffmpeg` trong `PATH`.

Ngoài ra cần Hugging Face token và đã accept terms cho:
- `pyannote/speaker-diarization-3.1`
- `pyannote/segmentation-3.0`
- `pyannote/embedding`

Token được đọc theo thứ tự:
1. `--hf-token`
2. `HF_TOKEN`
3. `HUGGINGFACE_TOKEN`
4. `HUGGINGFACE_ACCESS_TOKEN`
5. `hugging_face_key.txt`

## Chạy pipeline

### A) Audio-only (default mới)
```bash
python run_pipeline.py \
  --audio-dir outputs \
  --output-dir my_dataset \
  --dataset-name vn_voice \
  --label-csv data_label_by_hand.csv
```

Có thể tune diarization:
```bash
python run_pipeline.py \
  --audio-dir outputs \
  --audio-pattern "*.mp3,*.wav" \
  --device auto \
  --merge-gap 2.0 \
  --min-segment-duration 0.5 \
  --min-overlap 0.70 \
  --seg-min-duration-off 0.3 \
  --clustering-threshold 0.7
```

### B) Legacy config + transcript
```bash
python run_pipeline.py \
  --input-json config_test.json \
  --output-dir my_dataset \
  --dataset-name vn_voice \
  --label-csv data_label_by_hand.csv
```

## Output

Thư mục output:
- `wavs/`: file wav đã cắt (mono, 16k)
- `metadata.csv`
- `hf_dataset/`

### Schema metadata (audio-only mode)
- `segment_id`
- `audio`
- `duration`
- `start_sec`
- `end_sec`
- `abs_start_sec`
- `abs_end_sec`
- `source_file`
- `diarization_speaker`
- `speaker_label`
- `speaker_id`
- `speaker_name`
- `speaker_gender`
- `speaker_region`
- `speaker_position`
- `overlap_ratio`

Rule:
- Không dùng transcript trong audio-only mode.
- `speaker_label = speaker_name` nếu map CSV thành công, ngược lại dùng `diarization_speaker`.
- `speaker_id = 0` khi chưa map được speaker thật.

## File chính
- `run_pipeline.py`: entrypoint unified (legacy + audio-only)
- `src/pyannote_diarization.py`: pyannote diarization + merge segments
- `src/audio_only_dataset.py`: build dataset audio-only
- `edit_audio.py`: logic legacy
