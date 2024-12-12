import yt_dlp
import os
import warnings
from faster_whisper import WhisperModel

def transcribe_downloaded_audio():
    output_dir = "./transcriptions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    audio_files = os.listdir("./youtube")
    
    for audio in audio_files:
        audio_path = os.path.join("./youtube", audio)
        output_file = os.path.join(output_dir, f"{os.path.splitext(audio)[0]}.txt")
        breakpoint()
        try:
            text = transcribe_audio(audio_path)
            if not text:
                raise Exception("transcribe_downloaded_audio: text is None")
            
            with open(output_file, 'w') as f:
                count = 0
                for word in text.split():
                    if count + len(word) > 100:
                        f.write("\n")
                        count = 0
                    else:
                        f.write(" " if count > 0 else "")
                    f.write(word)
                    count += len(word) + 1
                f.write("\n")
            print(f"transcribe_downloaded_audio: Transcribed {audio} to {output_file}")
        except Exception as e:
            print(f"transcribe_downloaded_audio: Error transcribing {audio}: {str(e)}")

def download_youtube_audio(url, output_path='youtube'):
    """
    Download audio from YouTube video
    """
    
    print("Starting download...")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'cookiesfrombrowser': ('chrome',),  # Use cookies from Chrome browser
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            
            if not info:
                raise Exception("info is None")
            
            # Use the original title since that's what yt-dlp uses
            
            print("Finished download...")
            return os.path.join(output_path, f"{info['title']}.mp3")
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            return None

def transcribe_audio(audio_path):
    """
    Transcribe audio file to text using Faster Whisper
    """
    try:
        # Run on CPU with INT8 quantization
        model = WhisperModel("small", device="cpu", compute_type="int8")
        
        print("transcribe_audio: Transcribing audio...")
        # Transcribe audio with improved settings
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True
        )
        
        # Combine all segments into one text
        text = " ".join([segment.text for segment in segments])
        
        print("transcribe_audio: audio transcribed")
        return text
    except Exception as e:
        print(f"transcribe_audio: Error transcribing audio: {str(e)}")
        return None

def youtube_to_text(url):
    """
    Main function to convert YouTube video to text
    """
    print("youtube_to_text: Starting youtube to text conversion. ")
    audio_path = download_youtube_audio(url)
    
    if audio_path is None:
        print("youtube_to_text: Failed to download audio. Exiting.")
        return None


    absolute_path = os.path.abspath(audio_path)
    
    breakpoint()
    
    if os.path.exists(absolute_path):
        text = transcribe_audio(absolute_path)
        if text:
            print("youtube_to_text: Transcription complete!")
        return text
    else:
        print(f"youtube_to_text: Audio file not found: {absolute_path}")
        return None

if __name__ == "__main__":
    # Example usage
    # video_url = input("Enter YouTube URL: ")
    # text = youtube_to_text(video_url)
    # if text:
    #     print("\nTranscribed Text:")
    #     print(text)
    
    transcribe_downloaded_audio()
