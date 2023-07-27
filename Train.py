


from __future__ import print_function

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


from six.moves import xrange

import torch
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from dataLoader.DataLoader import myDataset
from model.model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




batch_size = 32
epoch = 1
save_iter = 5000
print_iter = 100
img_size = 96
num_training_updates = 25000
name = 'STL10'
train_data_path = r"E:\DataSet\OT_cut\STL\train"

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25
decay = 0.99
learning_rate = 1e-3


save_path = f'./checkpoints/{name}'
if not os.path.exists(save_path):
    os.makedirs(save_path)


### Load Data
transforms = transforms.Compose([
    # transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
])
training_data = myDataset(train_data_path, transform=transforms, img_size=img_size)
# training_data = datasets.CIFAR10(root=r"E:\DataSet\CIFAR10", train=True, download=True,
#                                   transform=transforms.Compose([
#                                       transforms.ToTensor(),
#                                       transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
#                                   ]))
data_variance = np.var(training_data.data / 255.0)
training_loader = DataLoader(training_data,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True)

print(f"training_loader:{len(training_loader)}")







### Define Model
model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim,
              commitment_cost, decay).to(device)


### Define Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

### Training
model.train()
train_res_recon_error = []
train_res_perplexity = []


for i in xrange(num_training_updates):
    (_, data) = next(iter(training_loader))
    # (data, _) = next(iter(training_loader))
    data = data.to(device)
    optimizer.zero_grad()

    vq_loss, data_recon, perplexity = model(data)
    recon_error = F.mse_loss(data_recon, data) / data_variance
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()

    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())

    if (i + 1) % save_iter == 0:
        # 保存模型
        torch.save(model.state_dict(), os.path.join(save_path, f'{name}_vqvae_{i + 1}.pth'))
    if (i + 1) % print_iter == 0:
        print('%d iterations' % (i + 1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        print()


train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)

# In[18]:


f = plt.figure(figsize=(16, 8))
ax = f.add_subplot(1, 2, 1)
ax.plot(train_res_recon_error_smooth)
ax.set_yscale('log')
ax.set_title('Smoothed NMSE.')
ax.set_xlabel('iteration')

ax = f.add_subplot(1, 2, 2)
ax.plot(train_res_perplexity_smooth)
ax.set_title('Smoothed Average codebook usage (perplexity).')
ax.set_xlabel('iteration')

plt.savefig(os.path.join(save_path, f'{name}_vqvae_{i + 1}.png'))
plt.show()

