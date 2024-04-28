import streamlit as st
from pytube import YouTube
import whisper
import os
from pathlib import Path
from zipfile import ZipFile
import shutil
from dotenv import find_dotenv, load_dotenv
from langchain_groq import ChatGroq

dotenv_path = '.env'
load_dotenv(dotenv_path)
GROQ_API = os.getenv('GROQ_API')


@st.cache
def load_whisper():
    model = whisper.load_model(name="base")
    return model

def save_audio(url:str):
    youtube = YouTube(url)
    audio = youtube.streams.filter(only_audio=True).first()
    audio_file = audio.download()
    base, ext = os.path.splitext(audio_file)    # splits a file path into a pair (root, ext). The root is everything leading up to the last dot, and the ext is everything after the last dot.
    audio_file_path = base + '.mp3'
    try:
        os.rename(audio_file, audio_file_path)
    except Exception:
        os.remove(audio_file_path)
        os.rename(audio_file, audio_file_path)
    audio_file_name = Path(audio_file_path).stem + '.mp3' #  returns the stem of a file path, which is the filename without the extension.
    print(f'{youtube.title} has successfully downloaded!')
    print(audio_file_name)
    return youtube.title, audio_file_name

def audio_to_transcript(audio_file):
    model = load_whisper()
    reult = model.transcribe(audio_file, language="en", task="transcribe")
    transcript = reult['text']
    return transcript

def transcript_article(text):
    
    from langchain_core.prompts import ChatPromptTemplate
    chat = ChatGroq(temperature=0.5,
                    model_name="Llama3-70b-8192",
                    api_key=GROQ_API,
                    max_tokens=100,
                    model_kwargs={
                        "top_p": 1,
                        "frequency_penalty": 0.0,
                        "presence_penalty" : 0.0
                    }
                    )
    system = "Generate News Article based on the Transcript Provided?"
    human = f'{text}'
    prompt = ChatPromptTemplate.from_messages(messages=[('system', system), ('human', human)])
    chain = prompt | chat
    response = chain.invoke({"query": f'{text}'})
    return response

# stremlit UI
st.markdown('# **News Article Generator App from YouTube Videos**')
st.header('Input the YouTube Video url')
url = st.text_input('Enter the video url here')
if st.checkbox('Start Analysis...'):
    title, audio_file_name = save_audio(url)
    st.audio(audio_file_name)
    transcript = audio_to_transcript(audio_file_name)
    st.header('Transcript are being generated')
    st.success(transcript)
    article = transcript_article(text=transcript_article)
    st.success(article)

    # write transcript to a transcript.txt
    with open('transcript.txt', 'w') as transcript_file:
        transcript_file.writ(transcript)
    # write article to article.txt
    with open('article.txt', 'w') as article_file:
        article_file.write(article)
    
    with ZipFile('output.zip', 'w') as zip_file:
        zip_file.write('transcript.txt')
        zip_file.write('article.text')
    
    with open('output.zip', 'rb') as zip_download:
        btn = st.button(
            label='Download Zip',
            data=zip_download,
            file_name='output.zip',
            mime='application/zip'
        )
