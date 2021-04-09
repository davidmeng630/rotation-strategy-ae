import torch.nn as nn
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation

class RotationStrategyAutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoding_hidden_layer1 = nn.Linear(in_features=kwargs['input_size'], out_features=64)
        self.encoding_hidden_layer2 = nn.Linear(in_features=64, out_features=32)
        self.core_hidden_layer = nn.Linear(in_features=32, out_features=3)
        self.decoding_hidden_layer2 = nn.Linear(in_features=3, out_features=32)
        self.decoding_hidden_layer1 = nn.Linear(in_features=32, out_features=64)
        self.output_layer = nn.Linear(in_features=64, out_features=kwargs['output_size'])

    def forward(self, features):
        layer = self.encoding_hidden_layer1(features)
        layer = torch.relu(layer)
        layer = self.encoding_hidden_layer2(layer)
        layer = torch.relu(layer)
        layer = self.core_hidden_layer(layer)
        layer = torch.relu(layer)
        layer = self.decoding_hidden_layer2(layer)
        layer = torch.relu(layer)
        layer = self.decoding_hidden_layer1(layer)
        layer = torch.relu(layer)
        layer = self.output_layer(layer)
        layer = torch.relu(layer)  # reconstruction by AE
        return layer

class RotationStrategyTrainedEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoding_hidden_layer1 = nn.Linear(in_features=kwargs['input_size'], out_features=64)
        self.encoding_hidden_layer2 = nn.Linear(in_features=64, out_features=32)
        self.core_hidden_layer = nn.Linear(in_features=32, out_features=3)

    def forward(self, features):
        layer = self.encoding_hidden_layer1(features)
        layer = torch.relu(layer)
        layer = self.encoding_hidden_layer2(layer)
        layer = torch.relu(layer)
        layer = self.core_hidden_layer(layer)
        layer = torch.relu(layer)
        return layer

class SectorEtfDataset(Dataset):

    def __init__(self, parquet_file, transform = None):
        self.data = pd.read_parquet(parquet_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data.iloc[idx, :-1].astype('float32')
        if self.transform:
            item = self.transform(item)
        return item

class ToTensor(object):

    def __call__(self, series):
        return torch.from_numpy(series.to_numpy())

etf_dataset = SectorEtfDataset('./data/sector_etf_log_returns.parquet', ToTensor())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
saved_path = './model/rotation_strategy.pth'

# training the model
"""
model = RotationStrategyAutoEncoder(input_size=22, output_size=22).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
batch_size = 16

dataloader = DataLoader(etf_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

epochs = 50
for epoch in range(epochs):
    loss = 0
    for batch_features in dataloader:
        batch_features = batch_features.to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = criterion(outputs, batch_features)

        # compute gradients backward
        train_loss.backward()
        # perform parameter update based on current gradients
        optimizer.step()
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

     # compute the epoch training loss
    loss = loss / len(dataloader)
    print('epoch : {}/{}, loss = {:.6f}'.format(epoch+1, epochs, loss))

# save model
# torch.save(model.state_dict(), saved_path)
"""

encoder = RotationStrategyTrainedEncoder(input_size=22)
encoder.load_state_dict(torch.load(saved_path), strict=False)

arr = []
for features in etf_dataset:
    #print(features)
    features = features.to(device)
    outputs = encoder(features)
    print(outputs)
    arr.append(outputs.detach().tolist())

points = np.array(arr)
print(points.shape)
start = 0
end = 250
fig = plt.figure(figsize=(10,8))
ax = p3.Axes3D(fig)
ax.set_xlim(0,0.8)
ax.set_ylim(0,0.8)
ax.set_zlim(0,0.8)
#for idx in range(start, end):
#    ax.scatter(*points[idx], color=plt.cm.jet((float(idx)/float(end-start+1))), s=20, edgecolor='k')

def update(idx):
    scat = ax.scatter(*points[idx], color=plt.cm.jet((float(idx) / float(end-start+1))), s=50, edgecolor='k')
    return scat

animation = FuncAnimation(fig,  update, frames=(end-start+1), interval=100)
plt.title('Latent space dynamics')
plt.show()
