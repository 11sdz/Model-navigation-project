import yt_dlp

# URL of the YouTube video
video_url = ("https://www.youtube.com/watch?v=CZiN0hNW5B4")

# yt-dlp options for downloading 1080p video
ydl_opts = {
    'format': 'bestvideo[height=1080]+bestaudio/best',
    'merge_output_format': 'mp4',  # Ensures the final output is in MP4 format
    'outtmpl': '%(title)s.%(ext)s',  # Save with the video title as filename
    'noplaylist': True,  # Download only a single video
}

# Download video
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])

print("Download complete!")
