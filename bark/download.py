import glob
import json
import os
import requests


def get_dataset():

    files = []

    for d in ["2021","2022","2023","2024"]:
        files += glob.glob(f"/home/pkd/code/pst/raw/{d}/*.json")

    # for each file in files, load the json, exctract the text and the audio path
    # and store them in two lists

    text_files = []
    audio_files = []

    for file in files:
        with open(file) as f:
            data = json.load(f)
            # if file doesnt exist, download audio from urlS3
            audio_file = "../data/" + data["magnetothequeId"] + ".mp3"
            if not os.path.exists(audio_file):
                try:
                    url = data["urlS3"]
                except KeyError:
                    print(f"no urlS3 for {data['magnetothequeId']}")
                    continue
                r = requests.get(url, allow_redirects=True)
                open(audio_file, 'wb').write(r.content)

            
            audio_files.append(audio_file)

            # text needs to be extracted as it is in transcript format
            text_file = "../data/" + data["magnetothequeId"] + ".txt"

            if not os.path.exists(text_file):
                text = ""
                for line in data['transcript']:
                    text += line['text'] + " "
                with open(text_file, "w") as f:
                    f.write(text)
            
            text_files.append(text_file)


    return audio_files, text_files