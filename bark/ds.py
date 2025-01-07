from pathlib import Path
import json
import torchaudio
from dataclasses import dataclass
from typing import List, Dict
import torch
import os

@dataclass
class AudioSegment:
    audio_path: Path
    json_path: Path
    start: float
    end: float
    text: str

def find_matching_files(audio_dir: Path, json_root: Path) -> List[tuple[Path, Path]]:
    """Find all matching MP3/JSON pairs"""
    audio_files = list(audio_dir.glob("*.mp3"))

    # Create a mapping of IDs to JSON files
    json_files = {
        p.stem: p
        for p in json_root.rglob("*.json")
    }

    # Match them up
    pairs = []
    for audio_path in audio_files:
        audio_id = audio_path.stem
        if audio_id in json_files:
            pairs.append((audio_path, json_files[audio_id]))
        else:
            print(f"⚠️ No JSON found for {audio_id}")

    return pairs

def load_audio_ds(audio_dir: str, json_root: str, max_duration: float = 10.0) -> List[AudioSegment]:
    audio_path = Path(audio_dir)
    json_path = Path(json_root)

    pairs = find_matching_files(audio_path, json_path)

    segments = []
    for audio_file, json_file in pairs:
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)["transcript"]
                for segment in data:
                    # Skip segments longer than max_duration
                    if segment['end'] - segment['start'] <= max_duration:
                        segments.append(AudioSegment(
                            audio_path=audio_file,
                            json_path=json_file,
                            start=segment['start'],
                            end=segment['end'],
                            text=segment['text']
                        ))
            except json.JSONDecodeError:
                print(f"💀 Failed to parse {json_file}")

    return segments

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self,
                 audio_dir: str,
                 json_root: str,
                 sample_rate: int = 16000,
                 max_duration: float = 10.0):

        self.segments = load_audio_ds(audio_dir, json_root, max_duration)
        self.target_sr = sample_rate
        self.audio_cache: Dict[Path, torch.Tensor] = {}

    def load_audio_file(self, path: Path) -> torch.Tensor:

        if len(self.audio_cache) >= 5:
            self.audio_cache.clear()

        if path not in self.audio_cache:
            waveform, sr = torchaudio.load(path)
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform)
            self.audio_cache[path] = waveform
        return self.audio_cache[path]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]

        # Load and slice audio
        full_audio = self.load_audio_file(segment.audio_path)
        start_frame = int(segment.start * self.target_sr)
        end_frame = int(segment.end * self.target_sr)
        audio_segment = full_audio[:, start_frame:end_frame]

        return {
            'audio': audio_segment,
            'text': segment.text,
            'start': segment.start,
            'end': segment.end,
            'audio_path': str(segment.audio_path),
            'json_path': str(segment.json_path)
        }
    
    def get_no_audio(self,idx):
        segment = self.segments[idx]
        return {
            'text': segment.text,
            'start': segment.start,
            'end': segment.end,
            'audio_path': str(segment.audio_path),
            'json_path': str(segment.json_path)
        }


import pandas as pd
import huggingface_hub
from datasets import Dataset, Audio, Features, Value, load_dataset

def prepare_hf_dataset(
    audio_dir: str,
    json_root: str,
    output_dir: str,
    max_duration: float = 10.0
):
    # Create output directory with a cleaned audio subfolder
    output_path = Path(output_dir)
    audio_output = output_path / "audio"
    audio_output.mkdir(parents=True, exist_ok=True)

    # First, collect all data into a list of dicts
    dataset_items = []
    dataset = AudioDataset(audio_dir, json_root, max_duration=max_duration)

    print("🎵 Processing audio files...")
    
    
    for idx in range(len(dataset)):
        item = dataset.get_no_audio(idx)

        print(f"Processing {item['audio_path']}")

        # Create a unique filename for this segment
        audio_id = Path(item['audio_path']).stem
        segment_filename = f"{audio_id}_{item['start']:.3f}_{item['end']:.3f}.wav"
        output_audio_path = audio_output / segment_filename
    
        if output_audio_path.exists(): 
           dataset_items.append({
                'audio': str(output_audio_path),
                'text': item['text'],
                'start_time': item['start'],
                'end_time': item['end'],
                'original_audio': item['audio_path'],
                'source_json': item['json_path']
            })
        

    
    

    print("📦 Creating HF dataset...")

    # Convert to pandas then to HF dataset
    df = pd.DataFrame(dataset_items)
    hf_dataset = Dataset.from_pandas(df)

    # Add Audio feature
    hf_dataset = hf_dataset.cast_column("audio", Audio(sampling_rate=dataset.target_sr))

    # Save metadata
    metadata = {
        "sampling_rate": dataset.target_sr,
        "total_segments": len(dataset_items),
        "max_duration": max_duration,
        "original_audio_dir": audio_dir,
        "original_json_dir": json_root
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save the dataset
    hf_dataset.save_to_disk(output_path)

    print(f"✨ Dataset saved! Found {len(dataset_items)} segments")
    return hf_dataset

def upload_to_hf(
    dataset_path: str,
    hf_repo_id: str,
    private: bool = False
):
    
    huggingface_hub.HfApi().create_repo(
        repo_id=hf_repo_id,
        exist_ok=True,
        private=private,
        token=os.getenv("HUGGINGFACE_TOKEN")
    )



    """Upload the dataset to HuggingFace"""
    dataset = load_dataset("audiofolder", data_dir=dataset_path)

    dataset.push_to_hub(
        hf_repo_id,
        private=private,
        token=os.getenv("HUGGINGFACE_TOKEN")
    )
    print(f"🚀 Dataset uploaded to {hf_repo_id}!")

if __name__ == "__main__":
    # First prepare the dataset
    #dataset = prepare_hf_dataset(
    #    audio_dir="../data",
    #    json_root="../../raw",
    #    output_dir="./prepared_dataset",
    #    max_duration=10.0
    #)

    # Then upload it (uncomment when ready)
    upload_to_hf(
        dataset_path="./prepared_dataset",
        hf_repo_id="pkd/pst-audio",
        private=False  # Set to True if you're feeling shy
    )