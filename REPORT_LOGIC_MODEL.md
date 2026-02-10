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
  - `--audio-dir` (batch audio) hoac `--audio-files` (list audio cu the)
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
- Neu cung `diarization_speaker` match nhieu `speaker_name`, chon `speaker_name` co tong overlap lon nhat (canonical mapping)

Fallback:
- Neu khong match duoc CSV timeline => giu `diarization_speaker`
- Ten speaker that (neu co) khong luu trong metadata ma luu file mapping rieng
- `speaker_id` duoc cap global theo toan bo batch audio

## 4) Output schema mode moi

- `segment_id`
- `audio`
- `duration`
- `start_sec`
- `end_sec`
- `start_sec_glob` (HH:MM:SS trong audio tong sau cong offset)
- `end_sec_glob` (HH:MM:SS trong audio tong sau cong offset)
- `source_file`
- `diarization_speaker`
- `speaker_id`
- `speaker_gender`
- `speaker_region`

Quy uoc:
- `segment_id = {audio_stem}_{index:04d}`
- `diarization_speaker = SPEAKER_XX+source_file_stem`
- `speaker_name_mapping.csv` luu cap (`diarization_speaker`, `speaker_name`) de doi chieu
- Neu khong xac dinh duoc `speaker_name` thi khong ghi dong mapping do vao CSV
- segment max `20s`, min `2.5s`, ngan hon bi loai

## 5) Tham so mac dinh quan trong

- `merge_gap = 2.0s`
- `min_segment_duration = 2.5s`
- `max_segment_duration = 20.0s`
- `min_overlap = 0.70`
- backend diarization: `pyannote/speaker-diarization-3.1`

## 6) Error handling

Mode moi canh bao ro:
- thieu `--audio-dir` va `--audio-files`
- thieu ffmpeg
- khong co file audio phu hop pattern
- thieu HF token / chua cai pyannote dependencies
- file audio loi, cat segment loi

## 7) Tong ket

Pipeline da duoc hop nhat vao `run_pipeline.py`:
- **new default**: audio-only end-to-end
- **legacy preserved**: config/transcript path khong bi pha vo
