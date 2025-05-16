from flask import Blueprint, request
from flask_cors import CORS
import os
import re
import time
import glob
import requests
import json
from tqdm import tqdm
from yt_dlp import YoutubeDL
import math
import subprocess
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
import base64
import shutil


worker = Blueprint('worker', __name__)
CORS(worker)

all_data = {}
SEGMENT_DURATION = 30

def download_and_split_video(url, output_dir="video_segments"):
    """
    Downloads a YouTube video and splits it into 30-second segments.
    
    Args:
        url (str): YouTube video URL
        output_dir (str): Directory to store the video segments
    """
    youtube_id = url.split("watch?v=")[1]

    def clear_directory_all(dir_path):
        for entry in os.listdir(dir_path):
            entry_path = os.path.join(dir_path, entry)
            try:
                if os.path.isfile(entry_path) or os.path.islink(entry_path):
                    os.remove(entry_path)
                elif os.path.isdir(entry_path):
                    shutil.rmtree(entry_path)
            except Exception as e:
                print(f"Failed to delete {entry_path}: {e}")
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        clear_directory_all(output_dir)
        
        # Configure youtube-dl options to download the best video quality
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': f'{youtube_id}_temp_video.%(ext)s'
        }
        
        # Download the video
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            duration = info['duration']  # Get video duration in seconds
            
        # Calculate the number of segments
        num_segments = math.ceil(duration / SEGMENT_DURATION)

        input_files = glob.glob(f'{youtube_id}_temp_video.*')
        input_file = input_files[0]
        
        # Split the video into segments
        for i in tqdm(range(num_segments), "Split the video into segments"):
            start_time = i * SEGMENT_DURATION
            output_file = os.path.join(output_dir, f'segment_{i:03d}.mp4')
            
            # Convert and split to mp4 format (force re-encoding to mp4)
            cmd = [
                'ffmpeg',
                '-i', input_file,  # Input format
                '-ss', str(start_time),
                '-t', str(SEGMENT_DURATION),
                '-c:v', 'mpeg4',  # Re-encode video using mpeg4 codec
                '-c:a', 'aac',  # Re-encode audio using AAC codec
                '-strict', 'experimental',  # Allow experimental codecs if needed
                output_file,
                '-y'  # Overwrite output files if they exist
            ]
            
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Clean up temporary file
        os.remove(input_file)
        
        print(f"Successfully split video into {num_segments} segments in '{output_dir}' directory")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def download_and_split_audio(url, output_dir="audio_segments"):
    """
    Downloads audio from a YouTube video and splits it into 30-second segments.
    
    Args:
        url (str): YouTube video URL
        output_dir (str): Directory to store the audio segments
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # remove old files
        files = glob.glob(output_dir + "/*")
        for file in files:
            print(file)
            os.remove(file)

        
        # Configure youtube-dl options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': 'temp_audio.%(ext)s'
        }
        
        # Download the audio
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            duration = info['duration']  # Get video duration in seconds
            
        # Calculate number of segments
        num_segments = math.ceil(duration / SEGMENT_DURATION)
        
        # Split the audio into segments
        for i in range(num_segments):
            start_time = i * SEGMENT_DURATION
            output_file = os.path.join(output_dir, f'segment_{i:03d}.mp3')
            
            cmd = [
                'ffmpeg',
                '-i', 'temp_audio.mp3',
                '-ss', str(start_time),
                '-t', str(SEGMENT_DURATION),
                '-c', 'copy',
                output_file,
                '-y'  # Overwrite output files if they exist
            ]
            
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
        # Clean up temporary file
        os.remove('temp_audio.mp3')
        
        print(f"Successfully split audio into {num_segments} segments in '{output_dir}' directory")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def find_files_by_ext(ext, base_path):
    found_files = []
    for root, dirs, files in os.walk(base_path):
        path = root.split(os.sep)
        for file in files:
            joined_path = "/".join(path) + "/"
            if file.split(".")[-1] == ext:
                found_files.append(joined_path + file)
    return found_files

def file_to_base64_binary(file_path: str):
    # assert file_path.lower().endswith(".mp4", ".jpg", ".jpeg", ".png", )
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")

def call_nvila_with_filepath(model:str="NVILA-15B", filepath=None):
    assert filepath
    client = OpenAI(
        base_url="http://localhost:8000",
        api_key="fake-key",
    )
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": "data:video/mp4;base64,{}".format(file_to_base64_binary(filepath)),
                        },
                        "frames": 16,
                    },
                    {"type": "text", "text": "Please describe the video"},
                ],
            }
        ],
        model=model
    )

    return response.choices[0].message.content

def narrate(input_dir):
    narrative = []
    all_files = find_files_by_ext("mp4", input_dir)
    all_files = sorted(all_files)
    timestamp = 0
    for i, file in tqdm(enumerate(all_files), "Narrate video segment"):
        print("Starting ", i, " out of ", len(all_files))
        narrative.append({"start": timestamp, "end": timestamp+SEGMENT_DURATION, "narrative": call_nvila_with_filepath(filepath=file)})
        print(f"\tNarration Complete")
        timestamp += SEGMENT_DURATION

    return narrative

def transcribe(output_dir):
    transcript = []
    all_files = find_files_by_ext("mp3", output_dir)
    all_files = sorted(all_files)
    timestamp = 0
    for i,file in enumerate(all_files):
        print("Starting ", i, " out of ", len(all_files))
        files = {'file': open(file, 'rb')}
        resp = requests.post("http://localhost:9303/transcription/transcribe", files=files)
        resp_json = resp.json()
        print(f"\tTranscription Complete")

        if resp.status_code == 200:
            output = resp_json.get("output", {"text": "", "chunks": []})
            for chunk in output["chunks"]:
                chunk["start"] = float(chunk["start"]) + timestamp
                chunk["end"] = float(chunk["end"]) + timestamp
            transcript.append(output)

        timestamp += SEGMENT_DURATION

    return transcript


def parse_pdf(pdf_url):
    response = requests.get(pdf_url, timeout=2)
    with open('paper.pdf', 'wb') as f:
        f.write(response.content)
    reader = PdfReader("paper.pdf")
    pdf_text = []
    for page in reader.pages:
        text = page.extract_text()
        pdf_text.append(text)
    return pdf_text

@worker.route('/narrateYouTube', methods=["GET"])
def startNarrateService():
    youtube_url = request.args.get("youtubeURL")

    # get the youtube transcript
    download_and_split_video(youtube_url, output_dir="video_segments")
    narrative = narrate("video_segments")
    shutil.rmtree("video_segments")

    return {"message": narrative}, 200


@worker.route('/transcribeYouTube', methods=["GET"])
def startTranscribeService():
    youtube_url = request.args.get("youtubeURL")

    if youtube_url in all_data:
        return all_data[youtube_url], 200
    else:
        all_data[youtube_url] = {
            "transcript": []
        }

        # get the youtube transcript
        download_and_split_audio(youtube_url, output_dir="audio_segments")
        raw_transcript = transcribe("audio_segments")
        all_data[youtube_url]["transcript"] = raw_transcript


        json.dump(all_data, open("all_data.json", "w"))


    return {"message": all_data[youtube_url]}, 200

@worker.route('/scrapePDF', methods=["GET"])
def startPDFScrape():
    pdf_url = request.args.get("pdfURL")

    if pdf_url in all_data:
        return all_data[pdf_url], 200
    else:
        all_data[pdf_url] = {
            "pdf_text": []
        }

    # get the pdf text
    pdf_text = parse_pdf(pdf_url)
    all_data[pdf_url]["pdf_text"] = pdf_text
    

    json.dump(all_data, open("all_data.json", "w"))


    return {"message": all_data[pdf_url]}, 200

if __name__ == "__main__":
    CURRENT_DIR = "/home/key4/JIRArena-exp"
    def load_jsonl(filepath):
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    data.append(json.loads(line))
        return data
    json_list = load_jsonl('/home/key4/JIRArena-exp/data/metainfo/filtered/entries_lectures.jsonl')
    for json_obj in json_list:
        video_link = json_obj["url"]
        youtube_id = video_link.split("watch?v=")[1]
        print(f"Start video {youtube_id}")
        temp_video_segments_dir = os.path.join(CURRENT_DIR, youtube_id)
        output_filepath = os.path.join(CURRENT_DIR, f"video_narratives/{youtube_id}.json")
        if os.path.isdir(temp_video_segments_dir) or os.path.exists(output_filepath):
            continue
        download_and_split_video(video_link, output_dir=temp_video_segments_dir)
        narrative = narrate(temp_video_segments_dir)
        if len(narrative) > 0:
            with open(output_filepath, "w") as file:
                json.dump(narrative, file)
        shutil.rmtree(temp_video_segments_dir)