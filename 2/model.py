from env import *
from torch.utils.data import Dataset, DataLoader
from torch import nn

class LanguageModelDataset(Dataset):
    def __init__(self, data:list[torch.Tensor]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]

    @staticmethod
    def get_dataloader(data:list[torch.Tensor], batch_size:int):
        dataset = LanguageModelDataset(data)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: (
                nn.utils.rnn.pad_sequence([s[0] for s in x], batch_first=True, padding_value=TOK_PAD),
                nn.utils.rnn.pad_sequence([s[1] for s in x], batch_first=True, padding_value=TOK_PAD),
            ),
        )
        return dataloader


class LSTMModel(nn.Module):
    def __init__(self, embed_dim:int, hidden_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(DICT_COUNT, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.lm_head = nn.Linear(hidden_dim, DICT_COUNT)

    def forward(self, x):
        x = self.embedding(x) 
        output, _ = self.lstm(x)
        output = self.lm_head(output)
        return output

class RNNModel(nn.Module):
    def __init__(self, embed_dim:int, hidden_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(DICT_COUNT, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.lm_head = nn.Linear(hidden_dim, DICT_COUNT)

    def forward(self, x):
        x = self.embedding(x) 
        output, _ = self.rnn(x)
        output = self.lm_head(output)
        return output
    
class FNNModel(nn.Module):
    def __init__(self, embed_dim:int, hidden_dim:int, window_size:int):
        super().__init__()
        self.window_size = window_size
        self.embedding = nn.Embedding(DICT_COUNT, embed_dim)
        self.fnn = nn.Sequential(
            nn.Linear(embed_dim*window_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, DICT_COUNT),
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.unfold(1, self.window_size, 1).transpose(2,3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        output = self.fnn(x)
        return output