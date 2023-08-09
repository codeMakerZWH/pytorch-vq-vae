import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os

import matplotlib.pyplot as plt
import numpy as np
import umap
from model.model import Model

import torch

from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from dataLoader.DataLoader import myDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 256
name = 'Carton'
which_epoch = 7500
img_size = 128


num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25
decay = 0.99

save_path = f'./checkpoints/{name}'

data_path = r'E:\DataSet\OT\Carton\valid'

transforms = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
])
validation_data = myDataset(data_path, transform=transforms, img_size=img_size,isTrain=False)
validation_loader = DataLoader(validation_data,
                               batch_size=batch_size,
                               shuffle=True,
                               pin_memory=True)
print(f"validation_loader:{len(validation_loader)}")


model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim,
              commitment_cost, decay).to(device)


model.load_state_dict(torch.load(os.path.join(save_path,f'{name}_vqvae_{which_epoch}.pth'), map_location=device))
# 将模型设置为评估模式
model.eval()

# 初始化计算指标的变量
total_psnr = 0.0
total_ssim = 0.0
num_samples = len(validation_loader.dataset)

with torch.no_grad():
    for valid_originals, _ in validation_loader:
        valid_originals = valid_originals.to(device)

        # 计算图像重建
        vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
        _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
        valid_reconstructions = model._decoder(valid_quantize)


        # 将数据从张量转换为 numpy 数组，并将通道放到最后一位
        valid_originals = np.moveaxis(valid_originals.cpu().numpy(), 1, -1)
        valid_reconstructions = np.moveaxis(valid_reconstructions.cpu().numpy(), 1, -1)

        # 计算 PSNR 和 SSIM
        batch_psnr = 0.0
        batch_ssim = 0.0
        for i in range(valid_originals.shape[0]):
            # For color images, set multichannel=True
            batch_psnr += psnr(valid_originals[i], valid_reconstructions[i], data_range=1.0)
            batch_ssim += ssim(valid_originals[i], valid_reconstructions[i], data_range=1.0, multichannel=True)

        # 累积批次结果
        total_psnr += batch_psnr
        total_ssim += batch_ssim

# 计算平均值
mean_psnr = total_psnr / num_samples
mean_ssim = total_ssim / num_samples

# 打印结果
print("Mean PSNR on validation set: {:.2f} dB".format(mean_psnr))
print("Mean SSIM on validation set: {:.4f}".format(mean_ssim))

(valid_originals, _) = next(iter(validation_loader))
valid_originals = valid_originals.to(device)

vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
_, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
valid_reconstructions = model._decoder(valid_quantize)



def show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()

def show_images(images, path, nrow=4, ncol=3):
    # Select a subset of images from the batch
    num_images = nrow * ncol
    selected_images = images[:num_images]

    # Reshape images for visualization (assuming images are of size [C, H, W])
    selected_images = selected_images.cpu().data
    selected_images = selected_images[:, :, :, :]  # Optional, only if the batch size is not equal to nrow * ncol

    selected_images = (selected_images+0.5)
    min_val = selected_images.min()
    max_val = selected_images.max()
    # Rescale the pixel values from [min_val, max_val] to [0, 1]
    selected_images = (selected_images - min_val) / (max_val - min_val)
    # Create a grid of images and display
    grid = make_grid(selected_images, nrow=nrow, ncol=ncol)
    npimg = grid.numpy()
    fig = plt.figure(figsize=(12, 12))  # Set the figsize to make the displayed image larger
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')  # Turn off axis labels
    # plt.savefig(path)
    # plt.close(fig)
    plt.show()
show_images(valid_reconstructions, path=f'1')
show_images(valid_originals, path=f'1')


# show(make_grid(valid_reconstructions.cpu().data)+0.5, )
# show(make_grid(valid_originals.cpu()+0.5))

proj = umap.UMAP(n_neighbors=3,
                 min_dist=0.1,
                 metric='cosine').fit_transform(model._vq_vae._embedding.weight.data.cpu())

plt.scatter(proj[:,0], proj[:,1], alpha=0.3)
plt.show()


