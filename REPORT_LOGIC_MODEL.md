# Bao Cao Logic Pipeline Unified

## 1) Muc tieu he thong

- Xay dung **mot command duy nhat** tao dataset tu audio dau vao.
- Bo qua transcript o mode moi, chi can:
  - segment
  - audio cut
  - speaker information
- Van giu mode legacy (`--input-json`) de tuong thich quy trinh cu.

## 2) Hai che do xu ly

### A. Audio-only mode (mac dinh moi)
- Trigger: khong truyen `--input-json`.
- Input:
  - `--audio-dir` (batch audio)
  - `--label-csv` (optional)
  - HF token (CLI/env/file)
- Core:
  1. diarization pyannote
  2. merge segment
  3. map speaker theo CSV timeline
  4. cut wav + build HF dataset
- Output:
  - `wavs/`
  - `metadata.csv`
  - `hf_dataset/`

### B. Legacy mode
- Trigger: co `--input-json`.
- Logic giu nguyen theo `config + transcript + segment_original`.

## 3) Mo hinh du lieu (audio-only)

Cho moi segment:
- `start_sec`, `end_sec`: moc thoi gian trong file part
- `offset(file_part)`: tong duration cac part truoc no trong cung nhom
- `abs_start_sec = start_sec + offset(file_part)`
- `abs_end_sec = end_sec + offset(file_part)`

Speaker matching:
- `overlap_ratio = intersection([abs_start_sec, abs_end_sec], [spk_start, spk_end]) / segment_duration`
- Match neu `overlap_ratio >= min_overlap` (default `0.70`)

Fallback:
- Neu khong match duoc CSV timeline => giu `diarization_speaker`
- `speaker_id = 0`, `speaker_name = ""`

## 4) Output schema mode moi

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

Quy uoc:
- `segment_id = {audio_stem}_{index:04d}`
- `speaker_label = speaker_name` neu match CSV thanh cong, nguoc lai `diarization_speaker`

## 5) Tham so mac dinh quan trong

- `merge_gap = 2.0s`
- `min_segment_duration = 0.5s`
- `min_overlap = 0.70`
- backend diarization: `pyannote/speaker-diarization-3.1`

## 6) Error handling

Mode moi canh bao ro:
- thieu `--audio-dir`
- thieu ffmpeg
- khong co file audio phu hop pattern
- thieu HF token / chua cai pyannote dependencies
- file audio loi, cat segment loi

## 7) Tong ket

Pipeline da duoc hop nhat vao `run_pipeline.py`:
- **new default**: audio-only end-to-end
- **legacy preserved**: config/transcript path khong bi pha vo
