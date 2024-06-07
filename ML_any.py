#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from myPackage.my_module import *
from torchvision.transforms import ToTensor


# %%
data = load_samples('all_complex')

# %%
samps = [(x, y) for x, y in zip(data.Bins.values, data.OptimalState.values)]

#%%
samps2 = [(x, y) for x, y in zip(data.Bins.values, [a + b for a,b in zip(np.real(data.Matrix).tolist(),np.imag(data.Matrix).tolist())])]
# %%
import pickle
with open('ML_all_samples.dat', 'wb') as f:
    pickle.dump(samps, f)
# %%
import pickle
with open('ML_all_samples2.dat', 'wb') as f:
    pickle.dump(samps2, f)

#%%
import pickle
with open('ML_all_samples.dat', 'rb') as f:
    samps = pickle.load(f)

# %%
import pickle
with open('ML_all_samples2.dat', 'rb') as f:
    samps2 = pickle.load(f)

#%%
class all_dataset(Dataset):
    
    def __init__(self,samples):
        super(Dataset, self).__init__()
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        hist, params = self.samples[idx]
        hist = torch.from_numpy(hist).float()
        params = torch.Tensor(params).float()
        return hist, params
    
class Net(nn.Module):
    
    def __init__(self, input=100, output=2):
        super().__init__()
        
        self.input_layer = nn.Linear(input, 50)
        self.output_layer = nn.Linear(20, output)
        self.hidden_layers = nn.Sequential(
            nn.Linear(50, 100),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(100,50),
            nn.Sigmoid(),
            nn.Dropout(p=0.3),
            nn.Linear(50,70),
            nn.Sigmoid(),
            nn.Dropout(p=0.3),
            nn.Linear(70,20),
            nn.Sigmoid(),
            nn.Dropout(p=0.3),
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x   
        
#%%
#Trainig & validation loop
def fit(model, samples_train, samples_val, batch_size=10, lr=0.05, epochs=10, dataset = all_dataset, criterion=nn.MSELoss()):    
    from torchmetrics import MeanSquaredError
    data_loader_train = DataLoader(dataset(samples_train), shuffle=True, batch_size=batch_size)
    data_loader_val = DataLoader(dataset(samples_val), shuffle=True, batch_size=batch_size)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=0.5) 
    metrics = MeanSquaredError()

    for epoch in range(epochs):
        #Training
        model.train()
        for hist, params in data_loader_train:
            optimizer.zero_grad()
            pred = model(hist)
            loss = criterion(pred, params)
            loss.backward()
            optimizer.step()
            metrics(pred, params)
        train_loss = metrics.compute()
        metrics.reset()
        #Validation
        model.eval()
        with torch.no_grad():
            for hist, params in data_loader_val:
                pred = model(hist)
                metrics(pred, params)
        val_loss = metrics.compute()
        metrics.reset()
        
        print(f'Epoch {epoch+1}/{epochs}: training loss: {train_loss:.5f}, validation loss: {val_loss:.5f}')
    
    return model    
    
# %%
# Evaluation
def evaluate_model(model, samples_eval, batch_size=10, dataset=all_dataset):
    from torchmetrics import MeanSquaredError
    metrics = MeanSquaredError()
    data_loader = DataLoader(dataset(samples_eval), shuffle=True, batch_size=batch_size)
    model.eval()
    with torch.no_grad():
        for hist, params in data_loader:
            pred = model(hist)
            metrics(pred, params)
    return metrics.compute()

#%%
model = fit(Net(), samps[:200000], samps[200000:250000], epochs=10, batch_size=100)

# %%
model2 = fit(Net(output=32), samps2[:20000], samps2[20000:25000], epochs=10, batch_size=100)

# %%
model.eval()
model(torch.Tensor(samps[10][0]).float())
# %%
