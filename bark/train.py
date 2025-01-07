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


from datasets import load_dataset
    
def train_with_hf_dataset(
    hf_repo_id: str,
    batch_size: int = 2,
    epochs: int = 10,
    learning_rate: float = 3e-4
):
    # Load dataset from HuggingFace Hub
    dataset = load_dataset(hf_repo_id)

    # Assuming the dataset has a 'train' split
    train_dataset = dataset['train']

    # Define a DataLoader
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Initialize model configuration
    config = GPTConfig()
    config.n_layer = 12
    config.n_head = 12
    config.n_embd = 768
    config.input_vocab_size = 10_048
    config.output_vocab_size = 10_048
    config.block_size = 1024
    config.dropout = 0.0
    config.bias = True

    # Initialize model, optimizer, and loss function
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    train(epochs, model, dataloader, optimizer, criterion)

    # Save the trained model
    torch.save(model.state_dict(), "audio_train_hf.pt")

if __name__ == "__main__":

    # Train the model with the dataset
    train_with_hf_dataset(
        hf_repo_id="pkd/pst-audio",
        batch_size=2,
        epochs=10,
        learning_rate=3e-4
    )