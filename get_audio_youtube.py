import yt_dlp

def download_list_audio(url_list):
    # Cấu hình tối ưu cho audio dài
    ydl_opts = {
        'format': 'bestaudio/best',  # Lấy chất lượng âm thanh tốt nhất
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',   # Chuyển sang định dạng mp3
            'preferredquality': '192', # Chất lượng 192kbps
        }],
        'outtmpl': '/data/data_diarization/%(title)s.%(ext)s', # Lưu vào thư mục đích
        'ignoreerrors': True,         # Bỏ qua video bị lỗi (video riêng tư, bản quyền...)
        'quiet': False,               # Hiển thị tiến trình tải
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"Bắt đầu tải {len(url_list)} video...")
        ydl.download(url_list)
        print("--- Hoàn tất danh sách ---")

# --- DANH SÁCH URL CỦA BẠN ---
urls = [
    'https://www.youtube.com/watch?v=IopTGZdIqa4&t=5s ',
    'https://www.youtube.com/watch?v=uhdkj_Eer2w ',
    'https://www.youtube.com/watch?v=GOizWHLdMS4 ',
    'https://www.youtube.com/watch?v=uFmKki4PI84 ',
    'https://www.youtube.com/watch?v=WTz94aSJMIQ ',
    'https://www.youtube.com/watch?v=w3rswxJJbJQ ',
    'https://www.youtube.com/watch?v=zLA_7artjus ',
    
]

if __name__ == "__main__":
    download_list_audio(urls)
