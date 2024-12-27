import torchaudio
import torch
from encodec import EncodecModel

from transformers import AutoTokenizer

def prepare_audio(audio_path):
    # Load audio file
    wav, sr = torchaudio.load(audio_path)

    # Resample to 24kHz (EnCodec's required sample rate)
    if sr != 24000:
        resampler = torchaudio.transforms.Resample(sr, 24000)
        wav = resampler(wav)

    # Get EnCodec model
    encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz")

    # Convert to codes
    with torch.no_grad():
        encoded_frames = encodec_model.encode(wav)
        codes = encoded_frames[0][0].transpose(0, 1)

    return codes

def prepare_text(text):
    tokenizer = AutoTokenizer.from_pretrained("suno/bark")
    tokens = tokenizer.encode(text)
    return tokens

class BarkDataset(torch.utils.data.Dataset):
    def __init__(self, audio_files, texts):
        self.audio_files = audio_files
        self.texts = texts

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_tokens = prepare_audio(self.audio_files[idx])
        text_tokens = prepare_text(self.texts[idx])

        return {
            'input_ids': text_tokens,
            'audio_tokens': audio_tokens
        }

def collate_fn(batch):
    # Implement proper padding here
    # *winks* Don't forget to mask those padded tokens!
    return {
        'input_ids': torch.tensor([x['input_ids'] for x in batch]),
        'audio_tokens': torch.tensor([x['audio_tokens'] for x in batch])
    }
