import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 128

sharp_size = 128
blur_size = 128

num_epochs = 1000
save_epoch = 100
print_epoch = 10


num_hiddens = 128  ##最终输出通道数
num_residual_layers = 4 ##残差层数
num_residual_hiddens = 64  ##残差层输出通道数

num_embeddings = 512
embedding_dim = 64

commitment_cost = 0.1
decay = 0.99
learning_rate = 1e-4

from model.model import Model
VQVAE_path = r"E:\Python porject\pytorch-vq-vae\checkpoints\Carton\Carton_vqvae_7500.pth"
VQVAE_Model = Model(num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32,
              num_embeddings=num_embeddings, embedding_dim=embedding_dim,
              commitment_cost=0.25, decay=0.99).to(device)
VQVAE_Model.load_state_dict(torch.load(VQVAE_path, map_location=device))

from model.MyEncoder import Model,_transform, convert_weights

vision_width = 64

model = Model(vision_layers=[3,4,6,3], embed_dim=embedding_dim, num_embeddings=num_embeddings,
              commitment_cost=commitment_cost, decay=decay, image_resolution=blur_size,
              vision_width=vision_width).to(device)
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
import torch.nn as nn
loss_img = nn.CrossEntropyLoss()

train_data_path = r"E:\DataSet\OT\Carton\train"
### Load Data
from torchvision import transforms
sharp_transform = transforms.Compose([
    transforms.Resize((sharp_size,sharp_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
])
blur_transform = _transform(model.visual.input_resolution)
from dataLoader.step2_DataLoader import VQStep2Dataset
training_data = VQStep2Dataset(train_data_path, img_size=sharp_size,sharp_transform =sharp_transform, blur_transform=blur_transform)
training_loader = DataLoader(training_data,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True)

train_error = []

def train(epoch, training_loader, model, VQVAE_Model, save_path,total_loss=0.0, total_psnr=0.0, total_ssim=0.0):

    total_mse = 0.0
    total_similarity = 0.0
    total_vq = 0.0

    train_data_size = len(training_loader)
    train_data = tqdm(training_loader, total=train_data_size, initial=1, leave=False, desc=f"Epoch {epoch+1}/{num_epochs}")

    for sharp, blur in iter(train_data):
        sharp = sharp.to(device)
        blur = blur.to(device)
        optimizer.zero_grad()

        sharp_vq_output_eval = VQVAE_Model._pre_vq_conv(VQVAE_Model._encoder(sharp))
        _, sharp_quantize, _, encoding = VQVAE_Model._vq_vae(sharp_vq_output_eval)

        blur_vq_output_eval = model(blur)
        vq_loss, blur_quantize, x, _ = model._vq_vae(blur_vq_output_eval)


        similarity, _ = model.getSimilarity(blur_quantize, sharp_quantize.detach())
        ground_truth = torch.arange(len(sharp), dtype=torch.long, device=device)
        similarity_loss = loss_img(similarity, ground_truth)

        mse_loss = F.mse_loss(blur_vq_output_eval, sharp_vq_output_eval.detach())

        beta = 0.01
        loss = similarity_loss + vq_loss



        # loss = mse_loss + lambda_cosine * (1 - cosine_similarity)

        # loss = mse_loss + (1 - cosine_similarity).mean()

        # logits_per_image, logits_per_text = model.getSimilarity(blur_quantize, sharp_quantize)
        # ground_truth = torch.arange(len(blur), dtype=torch.long, device=device)
        #
        # loss = loss_img(logits_per_image, ground_truth)

        loss.backward()
        optimizer.step()
        # if device == "cpu":
        #     optimizer.step()
        # else:
        #     convert_models_to_fp32(model)
        #     optimizer.step()
        #     convert_weights(model)

        total_mse += mse_loss.item()
        total_similarity += similarity_loss.item()
        total_vq += vq_loss.item()
        total_loss += loss.item()

        if (epoch + 1) % print_epoch == 0:
            with torch.no_grad():
                data_recon = VQVAE_Model._decoder(blur_quantize)
                data_target = VQVAE_Model._decoder(sharp_quantize)
                # 将数据从张量转换为 numpy 数组，并将通道放到最后一位
                valid_originals = np.moveaxis(sharp.cpu().numpy(), 1, -1)
                valid_reconstructions = np.moveaxis(data_recon.cpu().numpy(), 1, -1)
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
    if total_psnr > 0.0:
        show_images(data_recon, path=f'{save_path}/{epoch + 1}.png')
        show_images(sharp, path=f'{save_path}/{epoch + 1}_GT.png')
        show_images(data_target, path=f'{save_path}/{epoch + 1}_target.png')
        mean_psnr = total_psnr / train_data_size / batch_size
        mean_ssim = total_ssim / train_data_size / batch_size
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / train_data_size / batch_size:.4f}, PSNR: {mean_psnr:.4f}, SSIM: {mean_ssim:.4f}")
    else:
        # print(f"\nEpoch {epoch + 1}/{num_epochs}, Loss: {total_loss / train_data_size / batch_size:.4f}, MSE: {total_mse / train_data_size / batch_size:.4f}, Similarity: {total_similarity / train_data_size / batch_size:.4f}, VQ: {total_vq / train_data_size / batch_size:.4f}")
        print(f"\nEpoch {epoch + 1}/{num_epochs}, Loss: {total_loss}, MSE: {total_mse}, Similarity: {total_similarity}, VQ: {total_vq} ")

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
    plt.savefig(path)
    plt.close(fig)

def makeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

if __name__ == '__main__':
    name = 'VQ-VQ_CLIP' + train_data_path.split('\\')[-2]
    model_save_path = f'./checkpoints/{name}'
    img_save_path = f'{model_save_path}/reconstructions'
    makeDir(model_save_path)
    makeDir(img_save_path)
    name = f"{name}_blurSize{blur_size}_ed{embedding_dim}_ne{num_embeddings}"


    for i in range(num_epochs):
        model.train()
        VQVAE_Model.eval()
        train(i, training_loader, model, VQVAE_Model,save_path=img_save_path)

        if (i+1) % save_epoch == 0:
            torch.save(model.state_dict(), f'{model_save_path}/{name}_{str(i+1)}.pth')




