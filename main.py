"""
This is the main user interface of the system. It builds a streamlit UI to input a video/audio link and
select the summarization method. It outputs the summary of the video/audio file.
"""

import sound_extraction, sound_transcription, summarization
import streamlit as st

def main():
    st.set_page_config(page_title="Youtube Insights", page_icon=":banana:")
    st.header("Youtube Insights :banana:")
    selected_method = st.radio("Summarization Method: ", ["MapReduce", "Stuffing", "Refine"])
    link = st.text_input("Video/ audio link")

    if link:
        st.write("Generating summary...")
        file_path, base_name = sound_extraction.extract_sound(link)
        transcript_path = sound_transcription.transcribe_data(file_path, base_name)

        if selected_method == "MapReduce":
            summary = summarization.map_reduce(transcript_path)
        elif selected_method == "Stuffing":
            summary = summarization.stuff(transcript_path)
        elif selected_method == "Refine":
            summary = summarization.refine(transcript_path)

        st.info(summary)

        # Wrtie data to the ouput file   
        transcription_path = f'summary_data/{base_name}_{selected_method}.txt'
        with open(transcription_path, 'w') as output_file:
            output_file.write(summary)


if __name__ == "__main__":
    main()