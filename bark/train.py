from encodec import EncodecModel
import funcy
import logging
import numpy as np
from scipy.special import softmax
import torch
import torch.nn.functional as F
from torch import nn
import tqdm
from transformers import BertTokenizer
from huggingface_hub import hf_hub_download

from model import GPTConfig, GPT
from model_fine import FineGPT, FineGPTConfig

from data import BarkDataset, collate_fn

from download import get_dataset


     

def train_step(model, batch, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    input_ids = batch['input_ids']
    targets = batch['targets']

    # Forward pass
    logits, _ = model(input_ids)
    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()
    


def train(epochs, model, dataloader, optimizer, criterion):

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            loss = train_step(model, batch, optimizer, criterion)
            total_loss += loss

        print(f"Epoch {epoch}: Avg loss = {total_loss/len(dataloader)}")


            

if __name__ == "__main__":

    audio_files, text_files = get_dataset()

    texts = (open(f).read() for f in text_files)

    dataset = BarkDataset(audio_files, texts)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    config = GPTConfig()
    config.n_layer = 12
    config.n_head = 12
    config.n_embd = 768
    config.input_vocab_size = 10_048
    config.output_vocab_size = 10_048
    config.block_size = 1024
    config.dropout = 0.0
    config.bias = True

    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 10


    train(epochs, model, dataloader, optimizer, criterion)

    torch.save(model.state_dict(), "audio_train.pt")