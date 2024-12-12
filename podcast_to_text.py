import requests
import os
from urllib.parse import urlparse, unquote
from faster_whisper import WhisperModel

def download_podcast(url):
    """
    Download podcast from URL
    """
    try:
        # Create downloads directory if it doesn't exist
        if not os.path.exists("downloads"):
            os.makedirs("downloads")

        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = unquote(os.path.basename(parsed_url.path)).split('?')[0]
        
        # Download path
        download_path = os.path.join("downloads", filename)
        
        # Don't download if file already exists
        if os.path.exists(download_path):
            print(f"File already exists: {download_path}")
            return download_path

        # Download the file
        print(f"Downloading podcast...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Save the file
        with open(download_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Download completed: {download_path}")
        return download_path

    except Exception as e:
        print(f"Error downloading podcast: {str(e)}")
        return None

def transcribe_audio(audio_path):
    """
    Transcribe audio file to text using Faster Whisper
    """
    try:
        print("Loading model...")
        # Run on CPU with INT8 quantization
        model = WhisperModel("small", device="cpu", compute_type="int8")
        
        print("Transcribing audio...")
        # Transcribe audio with improved settings
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True
        )
        
        # Combine all segments into one text
        text = " ".join([segment.text for segment in segments])
        return text

    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        return None

def save_transcript(text, original_filename):
    """
    Save transcript to file
    """
    try:
        # Create transcriptions directory if it doesn't exist
        if not os.path.exists("transcriptions"):
            os.makedirs("transcriptions")

        # Create transcript filename
        transcript_filename = os.path.splitext(original_filename)[0] + ".txt"
        transcript_path = os.path.join("transcriptions", transcript_filename)

        # Save transcript
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f"Transcript saved to: {transcript_path}")
        return transcript_path

    except Exception as e:
        print(f"Error saving transcript: {str(e)}")
        return None

def process_podcast(url):
    """
    Main function to process podcast: download, transcribe, and save
    """
    # Download podcast
    audio_path = download_podcast(url)
    if not audio_path:
        return

    # Transcribe audio
    text = transcribe_audio(audio_path)
    if not text:
        return

    # Save transcript
    filename = os.path.basename(audio_path)
    save_transcript(text, filename)

if __name__ == "__main__":
    # Example usage
    podcast_url = input("Enter podcast URL: ")
    process_podcast(podcast_url)
