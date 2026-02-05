# HƯỚNG DẪN THÊM THÔNG TIN SPEAKER VÀO AUDIO DATASET
## 1. TỔNG QUAN
Tôi cần thêm thông tin speaker (người nói) vào dataset audio đã được transcribe. Dataset có cấu trúc phân cấp:
```
Audio gốc (hatinh1, hatinh2,...) 
    -> Audio nhỏ (hatinh1_1, hatinh1_2,...) 
        -> Audio segments (hatinh1_1_0000, hatinh1_1_0001,...)
```
## 2. CÁC FILE DỮ LIỆU
### 2.1 File `data_label_by_hand.csv`
Chứa thông tin speaker đã được label thủ công với cột:
- **ID**: Định danh audio gốc (hatinh1, hatinh2, hatinh3, hatinh4, hatinh5)
- **Trình tự người nói**: Danh sách speakers theo format `{id}_{gender}_{region} ({name} - {position})`
- **Timestamp**: Các khoảng thời gian tương ứng với từng speaker trong audio GỐC
Ví dụ "Trình tự người nói":
```
1_male_central (Nguyễn Hồng Lĩnh)
2_male_central (Nguyễn Trọng Hiếu)
10_female_south (Nguyễn Thị Nguyệt - Giám đốc Sở Giáo dục và Đào tạo Hà Tĩnh)
```
Ví dụ "Timestamp" (tương ứng 1:1 với từng dòng speaker):
```
0:43 - 4:03
4:27 - 57:40
1:14:19 - 1:14:49
```
### 2.2 File `config.json` (hoặc config_sample.json, config_test.json)
Chứa thông tin segments cho mỗi audio nhỏ:
```json
{
    "outputs/hatinh1_1.mp3": {
        "segments": [
            ["00:00.00", "00:02.61"],
            ["00:04.65", "00:32.39"],
            ...
        ],
        "segment_original": [
            ["00:37.949", "00:40.514"],
            ["00:43.332", "01:11.041"],
            ...
        ],
        "transcript_path": "txt/hatinh1_1.txt"
    }
}
```
Trong đó:
- **segments**: Timestamp của từng segment SAU KHI thêm marker phân tách và merge (timestamp reset từ 0)
- **segment_original**: Timestamp của từng segment TRƯỚC KHI thêm marker (timestamp gốc trong audio nhỏ)
- `segments[i]` tương ứng 1:1 với `segment_original[i]`
### 2.3 File `metadata.csv` (output dataset)
Chứa thông tin mỗi audio segment:
- `audio_id`: ID segment (VD: hatinh1_1_0000)
- `audio_path`: Đường dẫn file WAV
- `transcript`: Nội dung phiên âm
- `duration`: Độ dài (giây)
## 3. VẤN ĐỀ CẦN GIẢI QUYẾT
Audio gốc bị tách thành nhiều audio nhỏ, mỗi audio nhỏ có timestamp riêng bắt đầu từ 0. Cần:
1. Map `segment_original` của audio nhỏ => timestamp trong audio GỐC
2. So sánh với timestamps trong `data_label_by_hand.csv` để gán speaker
## 4. QUY TRÌNH MAPPING (4 BƯỚC)
### Bước 1: Xác định offset của audio nhỏ trong audio gốc
```
Audio gốc: hatinh1 (tổng ~3h19p)
├── hatinh1_1: offset = 0:00 (bắt đầu từ đầu)
├── hatinh1_2: offset = thời điểm kết thúc của hatinh1_1 trong audio gốc
├── hatinh1_3: offset = thời điểm kết thúc của hatinh1_2 trong audio gốc
└── ...
```
### Bước 2: Map segment -> segment_original
Đã có sẵn trong config.json, quan hệ 1:1 theo index:
```
segments[i] -> segment_original[i]
```
### Bước 3: Tính timestamp trong audio gốc
```
segment_original_gốc_start = segment_original_start + offset_audio_nhỏ
segment_original_gốc_end = segment_original_end + offset_audio_nhỏ
```
Ví dụ với hatinh1_1 (offset = 0):
- segment_original = ["00:43.332", "01:11.041"]
- segment_original_gốc = ["00:43.332", "01:11.041"]
Ví dụ với hatinh1_2 (offset = 59:51.554):
- segment_original = ["00:05.000", "00:30.000"]
- segment_original_gốc = ["59:56.554", "1:00:21.554"]
### Bước 4: Match với speaker (threshold overlap >= 70%)
So sánh `segment_original_gốc` với timestamps trong `data_label_by_hand.csv`:
- Tính tỷ lệ overlap = (thời gian giao nhau) / (độ dài segment)
- Nếu overlap >= 70% => gán speaker đó cho segment
- Nếu nhiều speakers có overlap >= 70% => chọn overlap cao nhất
## 5. THUẬT TOÁN
```python
def parse_timestamp_to_seconds(ts):
    """Convert timestamp string to seconds
    Handles formats: MM:SS.mmm, M:SS, H:MM:SS
    """
    parts = ts.replace('-', '').strip().split(':')
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return 0
def calculate_overlap_ratio(seg_start, seg_end, speaker_start, speaker_end):
    """Tính tỷ lệ overlap của segment với khoảng thời gian speaker"""
    seg_duration = seg_end - seg_start
    if seg_duration <= 0:
        return 0
    
    overlap_start = max(seg_start, speaker_start)
    overlap_end = min(seg_end, speaker_end)
    overlap_duration = max(0, overlap_end - overlap_start)
    
    return overlap_duration / seg_duration
def find_speaker(segment_start, segment_end, speaker_data, threshold=0.70):
    """
    Tìm speaker có overlap >= threshold với segment
    
    Args:
        segment_start, segment_end: Timestamp segment trong audio gốc (giây)
        speaker_data: List of (speaker_info, start, end)
        threshold: Ngưỡng overlap tối thiểu (default 70%)
    
    Returns:
        speaker_info nếu tìm thấy, None nếu không
    """
    best_speaker = None
    best_ratio = 0
    
    for speaker_info, sp_start, sp_end in speaker_data:
        ratio = calculate_overlap_ratio(segment_start, segment_end, sp_start, sp_end)
        if ratio >= threshold and ratio > best_ratio:
            best_ratio = ratio
            best_speaker = speaker_info
    
    return best_speaker, best_ratio
def parse_speaker_info(speaker_str):
    """
    Parse speaker string: "1_male_central (Nguyễn Hồng Lĩnh)"
    Returns: dict with id, gender, region, name, position
    """
    import re
    match = re.match(r'(\d+)_(male|female)_(central|south)\s*\(([^)]+)\)', speaker_str.strip())
    if match:
        id_num, gender, region, name_pos = match.groups()
        # Split name and position if exists
        if ' - ' in name_pos:
            name, position = name_pos.split(' - ', 1)
        else:
            name, position = name_pos, None
        return {
            'speaker_id': int(id_num),
            'speaker_gender': gender,
            'speaker_region': region,
            'speaker_name': name.strip(),
            'speaker_position': position.strip() if position else None
        }
    return None
```
## 6. OUTPUT MONG MUỐN
Thêm các cột sau vào `metadata.csv`:
| Tên cột | Kiểu | Mô tả | Ví dụ |
|---------|------|-------|-------|
| speaker_id | int | ID speaker trong session | 1, 2, 10 |
| speaker_name | str | Tên người nói | Nguyễn Hồng Lĩnh |
| speaker_gender | str | Giới tính | male / female |
| speaker_region | str | Vùng miền giọng nói | central / south |
| speaker_position | str | Chức danh (nếu có) | Giám đốc Sở Tài chính |
| overlap_ratio | float | Tỷ lệ overlap với speaker | 0.85 |
## 7. LƯU Ý QUAN TRỌNG
1. **Offset audio nhỏ**: Cần tính chính xác offset của từng audio nhỏ trong audio gốc. Có thể lấy từ segment_original cuối cùng của audio nhỏ trước đó.
2. **Timestamp format**: 
   - data_label_by_hand.csv: `M:SS` hoặc `H:MM:SS` (VD: "0:43", "1:14:19")
   - config.json: `MM:SS.mmm` (VD: "00:43.332", "01:11.041")
   - Cần chuẩn hóa về giây trước khi tính toán
3. **Threshold 70%**: Segment phải nằm trong khoảng thời gian của speaker ít nhất 70% độ dài của nó
4. **Không match**: Nếu không có speaker nào có overlap >= 70%, để trống hoặc đánh dấu "unknown"
5. **Speakers lặp lại**: Một speaker có thể xuất hiện nhiều lần (VD: 1_male_central là chủ tọa thường xuyên nói)
6. **Segments và segment_original**: Luôn có cùng số lượng phần tử và tương ứng 1:1 theo index
## 8. VÍ DỤ MINH HỌA
Với audio segment `hatinh1_1_0001`:
1. Từ config.json: `segment_original[1] = ["00:43.332", "01:11.041"]`
2. Offset của hatinh1_1 = 0 (là audio nhỏ đầu tiên)
3. segment_original_gốc = [43.332s, 71.041s]
4. Từ data_label_by_hand.csv: Speaker "2_male_central (Nguyễn Trọng Hiếu)" có timestamp "4:27 - 57:40" = [267s, 3460s]
5. Tính overlap: 
   - Segment: 43.332s - 71.041s (duration = 27.709s)
   - Speaker: 267s - 3460s
   - Không có overlap (segment kết thúc trước khi speaker bắt đầu)
6. Kiểm tra speaker trước: "1_male_central (Nguyễn Hồng Lĩnh)" có timestamp "0:43 - 4:03" = [43s, 243s]
   - Overlap: max(43.332, 43) = 43.332s đến min(71.041, 243) = 71.041s
   - Overlap duration = 71.041 - 43.332 = 27.709s
   - Ratio = 27.709 / 27.709 = 100% >= 70% ✓
7. => Gán speaker: 1_male_central (Nguyễn Hồng Lĩnh)
