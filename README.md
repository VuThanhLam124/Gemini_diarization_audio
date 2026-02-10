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
  --min-segment-duration 2.5 \
  --max-segment-duration 20 \
  --min-overlap 0.70 \
  --seg-min-duration-off 0.3 \
  --clustering-threshold 0.7
```

Chon nhieu file cu the bang list:
```bash
python run_pipeline.py \
  --audio-dir outputs \
  --audio-files '["hatinh1_1.mp3","hatinh1_2.mp3"]' \
  --output-dir my_dataset \
  --dataset-name vn_voice \
  --label-csv data_label_by_hand.csv
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
- `speaker_name_mapping.csv`
- `hf_dataset/`

### Schema metadata (audio-only mode)
- `segment_id`
- `audio`
- `duration`
- `start_sec`
- `end_sec`
- `start_sec_glob` (HH:MM:SS trong file tong)
- `end_sec_glob` (HH:MM:SS trong file tong)
- `source_file`
- `diarization_speaker`
- `speaker_id`
- `speaker_gender`
- `speaker_region`

Rule:
- Không dùng transcript trong audio-only mode.
- `diarization_speaker` duoc barcode theo format `SPEAKER_XX+source_file_stem`.
- `speaker_id` la global ID tren toan bo audio truyen vao.
- Ten that cua speaker duoc tach rieng trong `speaker_name_mapping.csv` (`diarization_speaker`, `speaker_name`).
- Mapping speaker tu label tay duoc chuan hoa theo `diarization_speaker` (chon speaker_name co tong overlap cao nhat) de giam map sai.
- Segment duoc gioi han: `min 2.5s`, `max 20s`; phan ngan hon 2.5s se bi bo.
- Cột `audio` trong `hf_dataset` được cast `Audio(..., decode=False)` để tương thích Kaggle/torchcodec.

## File chính
- `run_pipeline.py`: entrypoint unified (legacy + audio-only)
- `src/pyannote_diarization.py`: pyannote diarization + merge segments
- `src/audio_only_dataset.py`: build dataset audio-only
- `edit_audio.py`: logic legacy
