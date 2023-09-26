"""
This script file contains the data extraction functionality
TODO: Try different streams for faster vs better inference by Whisper model
"""

from pytube import YouTube
import os
# from moviepy.editor import VideoFileClip

def extract_sound(link: str) -> str:
    """
    Extracts the audio/mp4 file from the provided link, 
    changes the extension to .mp3 and returns the new file path and the base name.
    """

    yt = YouTube(link)
    # for stream in yt.streams.filter(only_audio=True):
    #     print(stream)
    #     print("\n")

    # Returns absolute path
    video = yt.streams.filter(only_audio=True).first().download(output_path="sound_data")
    # Get file name from the absolute path
    file_name = os.path.basename(video)
    # Replace spaces with underscores to prevent file reading issues
    file_name = file_name.replace(" ", "_")
    # Change file extension to mp3
    base_name, _ = os.path.splitext(file_name)
    new_file_name = f'{base_name}.mp3' 
    file_path = f'sound_data/{new_file_name}'
    os.rename(video, file_path)
    
    return file_path, base_name

# def convert_sound(in_file: str, out_file: str):
#     video_clip = VideoFileClip(in_file)
#     audio_clip = video_clip.audio
#     audio_clip.write_audiofile(out_file)
#     audio_clip.close()
#     video_clip.close()