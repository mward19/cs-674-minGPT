##################################################
# DATASET                                        #
##################################################

import json
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from tokenizers import Tokenizer
import torch

class JSONLDataset(Dataset):
    def __init__(self, file_path, split='train', test_size=1000, window_size=2048):
        self.tokenizer = Tokenizer.from_pretrained("gpt2")
        self.window_size = window_size

        self.file_path = file_path
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        encoded_lines = [self.tokenizer.encode(json.loads(line)['text']) for line in lines]
        
        print(f'Length of lines: {len(lines)}')
        # prelength = len(lines)
        data = []
        tokenized_data = []
        for i, encoded_line in tqdm(enumerate(encoded_lines), total=len(encoded_lines), desc='Loading dataset'):
            try:
                # Processes the line and extracts the 'text' field
                # Skips short text entries
                if len(encoded_line) <= self.window_size:
                    continue
                for start_idx in range(0, len(encoded_line) - window_size):
                    encoded_window_ids = encoded_line.ids[start_idx:start_idx+window_size+1]
                    encoded_window_tokens = encoded_line.tokens[start_idx:start_idx+window_size+1]
                    data.append(encoded_window_tokens)
                    tokenized_data.append(encoded_window_ids)

            except json.JSONDecodeError:
                # Silently skip lines that fail to parse
                pass 

        # print(f'Length of data: {len(data) / prelength * 100:.2f}%')
        
        # Implements the train/test split
        if split == 'train':
            self.data = data[:-test_size]
            self.tokenized_data = torch.tensor(tokenized_data[:-test_size])
        else:
            self.data = data[-test_size:]
            self.tokenized_data = torch.tensor(tokenized_data[-test_size:])

    def __len__(self):
        # Returns the total number of items in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Returns the raw text at the specified index
        return self.tokenized_data[idx][:-1], self.tokenized_data[idx][1:]

    def get_vocab_size(self):
        # your code here
        return self.tokenizer.get_vocab_size()

    def get_block_size(self):
        # your code here
        return self.window_size

# file_path = '/nobackup/autodelete/usr/YOUR_RC_USERNAME/pile_data_10.jsonl'
# # file_path = 'pile_data_10_first_50000.jsonl'
# # file_path = '100.jsonl'

# # Initialize the dataset with a test size of 1000 lines
# print(f'Making dataset... {get_elapsed_time() / 60} minutes')
# train_dataset = JSONLDataset(file_path, split='train', test_size=10)
# print(len(train_dataset))