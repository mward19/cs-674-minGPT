##################################################
# DATASET                                        #
##################################################
import json
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import os
from tqdm.auto import tqdm

class JSONLDataset(Dataset):
    def __init__(self, file_path, split='train', test_size=1000, window_size=1024):
        self.file_path = file_path
        self.window_size = window_size
        self.split = split

        tokenizer_path = "./gpt2_tokenizer"
        if os.path.exists(tokenizer_path):
            self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.save_pretrained(tokenizer_path)

        # Compute offsets to quickly access lines with valid 'text' without loading everything
        print("Indexing file offsets...")
        self.offsets = []
        with open(file_path, 'r', encoding='utf-8') as f:
            offset = 0
            for line in f:
                try:
                    # Only keep lines that contain 'text'
                    if 'text' in json.loads(line):
                        self.offsets.append(offset)
                except (json.JSONDecodeError, TypeError):
                    pass  # skip malformed lines
                offset += len(line.encode('utf-8'))

        # Implements the train/test split
        if split == 'train':
            self.offsets = self.offsets[:-test_size]
        else:
            self.offsets = self.offsets[-test_size:]

    def __len__(self):
        # Returns the total number of items in the dataset
        return len(self.offsets)

    def __getitem__(self, idx):
        # Returns the raw text at the specified index, tokenized on-demand
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            text = json.loads(line)['text']  # safe because filtered in __init__

            encoded = self.tokenizer.encode(text)

            if len(encoded) < self.window_size + 1:
                # Pad if line too short
                encoded += [self.tokenizer.eos_token_id] * (self.window_size + 1 - len(encoded))

            # Random window for training
            start_idx = torch.randint(0, len(encoded) - self.window_size, (1,)).item()
            window = encoded[start_idx:start_idx+self.window_size+1]

            return torch.tensor(window[:-1], dtype=torch.long), torch.tensor(window[1:], dtype=torch.long)

    def get_vocab_size(self):
        # returns the vocabulary size of the tokenizer
        return self.tokenizer.vocab_size

    def get_block_size(self):
        # returns the model input block size
        return self.window_size
