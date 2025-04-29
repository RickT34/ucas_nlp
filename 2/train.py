from env import *
from model import *
import sys

BATCH_SIZE = 32
EPOCHS = 30
DIM_EMB = 32
DIM_HIDDEN = 64
WINDOW_SIZE = 5

corpus = torch.load(PATH_CORPUS)
for i in range(len(corpus)):
    corpus[i] = corpus[i].to(DEVICE)
data = LanguageModelDataset.get_dataloader(corpus, BATCH_SIZE)
criterion = nn.CrossEntropyLoss(ignore_index=TOK_PAD)

def train_model(model):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    is_FNN = isinstance(model, FNNModel)
    for epoch in range(EPOCHS):
        loss_sum = 0
        for x, y in tqdm(data):
            if is_FNN:
                y = y[:, model.window_size-1:]
            optimizer.zero_grad()
            output = model(x)
            y = y.reshape(-1)
            output = output.reshape(-1, output.shape[-1])
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"Epoch {epoch+1}: loss={loss_sum/len(data)}")

model_name = sys.argv[1]
if model_name == "FNN":
    model = FNNModel(DIM_EMB, DIM_HIDDEN, WINDOW_SIZE).to(DEVICE)
elif model_name == "LSTM":
    model = LSTMModel(DIM_EMB, DIM_HIDDEN).to(DEVICE)
elif model_name == "RNN":
    model = RNNModel(DIM_EMB, DIM_HIDDEN).to(DEVICE)
else:
    raise ValueError("Invalid model name")

print(f"Training {model_name} model...")
train_model(model)
torch.save(model.state_dict(), DIR_DATA / f"{model_name}.pth")
# FNN: loss=4.092
# LSTM: loss=3.946
