"""
This script file contains the transcription functionality of the extracted data
using OpenAI Whisper model. Currently the base.en model is used
TODO: Test with different model versions (speed vs quality)
"""

import whisper 

def transcribe_data(file_path: str, base_name: str) -> str:
    """
    Transcribes the audio file, saves it to the transcript_data folder
    and returns the file_path of the transcription.
    """

    model = whisper.load_model("base.en")
    result = model.transcribe(file_path) 
    transcription = result["text"]

    # Wrtie data to the ouput file   
    transcription_path = f'transcript_data/{base_name}.txt'
    with open(transcription_path, 'w') as output_file:
        output_file.write(transcription)

    return transcription_path

