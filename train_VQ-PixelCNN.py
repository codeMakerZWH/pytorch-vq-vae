import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from model.PixelCNN import PixelCNN, Codebook
from model.model import Model
from dataLoader.step2_DataLoader import myDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set hyperparameters
batch_size = 16
num_epochs = 500
blur_size = 128
learning_rate = 1e-3
embedding_dim = 64    # Dimension of each codebook embedding
num_embeddings = 512  # Number of codebook embeddings
save_epoch = 10

# Create the modified PixelCNN model
PixelCNN_Model = PixelCNN(in_channels=3, out_channels=embedding_dim, num_blocks=3, num_layers=3,
                          num_embeddings=num_embeddings, embedding_dim=embedding_dim).to(device)

optimizer = optim.Adam(PixelCNN_Model.parameters(), lr=learning_rate)
print(PixelCNN_Model)

sharp_size = 128
VQVAE_path = r"E:\Python porject\pytorch-vq-vae\checkpoints\Carton\Carton_2_vqvae_7500.pth"
VQVAE_Model = Model(num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32,
              num_embeddings=num_embeddings, embedding_dim=embedding_dim,
              commitment_cost=0.25, decay=0.99).to(device)
VQVAE_Model.load_state_dict(torch.load(VQVAE_path, map_location=device))




train_data_path = r"E:\DataSet\OT\Carton\train"
### Load Data
sharp_transform = transforms.Compose([
    transforms.Resize((sharp_size,sharp_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
])
blur_transform = transforms.Compose([
    transforms.Resize((blur_size,blur_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
])
training_data = myDataset(train_data_path, sharp_transform =sharp_transform, blur_transform=blur_transform)

training_loader = DataLoader(training_data,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True)


# Define a function to train the model
def train(epoch, training_loader, PixelCNN_Model, VQVAE_Model,total_loss=0.0):
    train_data_size = len(training_loader)
    train_data = tqdm(training_loader, total=train_data_size, initial=1, leave=False, desc=f"Epoch {epoch+1}/{num_epochs}")

    for sharp, blur in iter(train_data):
        sharp = sharp.to(device)
        blur = blur.to(device)
        optimizer.zero_grad()

        vq_output_eval = VQVAE_Model._pre_vq_conv(VQVAE_Model._encoder(sharp))
        _, sharp_quantize, _, _ = VQVAE_Model._vq_vae(vq_output_eval)

        out = PixelCNN_Model(blur)


        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/train_data_size:.4f}")





if __name__ == '__main__':
    name = train_data_path.split('\\')[-2]
    name = f"{name}_PixelCNN{blur_size}_ed{embedding_dim}_ne{num_embeddings}"


    for i in range(num_epochs):
        PixelCNN_Model.train()
        VQVAE_Model.eval()
        train(i, training_loader, PixelCNN_Model, VQVAE_Model)

        if (i+1) % save_epoch == 0:
            torch.save(PixelCNN_Model.state_dict(), f'checkpoint/{name}_{str(i+1)}.pth')




