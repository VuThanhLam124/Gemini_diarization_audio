# Unified Audio Dataset Pipeline

## Tong quan

`run_pipeline.py` hien co 2 mode:

1. **Audio-only mode (mac dinh moi)**
   - Input: folder audio (`--audio-dir`)
   - Tu diarization bang pyannote
   - Map speaker theo `data_label_by_hand.csv` (neu co)
   - Khong can transcript/config json

2. **Legacy mode**
   - Input: `--input-json config_*.json`
   - Su dung transcript marker + `segment_original`
   - Giu nguyen logic cu de backward compatible

## Kien truc

```mermaid
flowchart TD
    subgraph NewMode["Audio-only mode (default)"]
        A1[Audio Folder\n--audio-dir]
        A2[Optional Speaker CSV\n--label-csv]
        A3[HF Token\n--hf-token or env]

        B1[Discover audio files]
        B2[Pyannote diarization\npyannote/speaker-diarization-3.1]
        B3[Merge adjacent segments\nmerge_gap + min_duration]
        B4[Compute part offsets\nfrom real durations]
        B5[Match speaker by overlap\nthreshold min_overlap]
        B6[Cut wav segments\nmono 16k]
        B7[Build HF Dataset + metadata.csv]

        A1 --> B1 --> B2 --> B3 --> B6 --> B7
        A1 --> B4 --> B5
        A2 --> B5
        A3 --> B2
        B3 --> B5
    end

    subgraph LegacyMode["Legacy mode (--input-json)"]
        L1[Config JSON + transcript]
        L2[run_pipeline.py legacy path]
        L3[edit_audio.py::create_hf_dataset]
        L4[wavs + metadata.csv + hf_dataset]

        L1 --> L2 --> L3 --> L4
    end
```

## Audio-only data flow

1. `run_pipeline.py` chay mode moi khi **khong co** `--input-json`.
2. Doc audio tu `--audio-dir` theo `--audio-pattern`.
3. Khoi tao pyannote 1 lan, diarize tung file, merge segment.
4. Tinh `audio_offset_map` tu tong duration cac part cung group (`^(.+)_(\d+)$`).
5. Match speaker theo timeline CSV:
   - `abs_start = start_sec + offset`
   - `abs_end = end_sec + offset`
   - overlap >= `--min-overlap` (default 0.70)
6. Cat tung segment thanh `wavs/*.wav`.
7. Xuat `metadata.csv` schema toi gian + `hf_dataset/`.

## Command

### Audio-only (default)
```bash
python run_pipeline.py \
  --audio-dir outputs \
  --output-dir my_dataset \
  --dataset-name vn_voice \
  --label-csv data_label_by_hand.csv
```

### Legacy
```bash
python run_pipeline.py \
  --input-json config_test.json \
  --output-dir my_dataset \
  --dataset-name vn_voice \
  --label-csv data_label_by_hand.csv
```

## Output schema (audio-only mode)

`metadata.csv` / HF dataset columns:

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

Rules:
- Khong su dung transcript trong mode moi.
- `speaker_label = speaker_name` neu map CSV thanh cong; nguoc lai la nhan diarization.
- `speaker_id = 0` neu unknown.
- `segment_id = {audio_stem}_{index:04d}`.

## File chinh

- `run_pipeline.py`: entrypoint unified (legacy + audio-only)
- `src/pyannote_diarization.py`: token, convert wav 16k, pyannote diarization, merge segment
- `src/audio_only_dataset.py`: batch process audio-only, map speaker, export metadata/HF
- `edit_audio.py`: logic legacy
