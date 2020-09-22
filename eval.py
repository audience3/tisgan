import torch
import numpy as np
import PIL
import XuNet
import model
import torch.nn as nn
from matplotlib import pyplot as plt
import torch.utils.data
import torchvision.utils as vutils
import torch.optim as optim
from PIL import Image
import cv2
from torchvision import transforms, datasets
import pytorch_ssim
import pytorch_msssim
import argparse

######################################################################

# make a parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--epoch', default=30,
                    type=int)  # the stratpoint means the currently epoch,and load the last epoch data
parser.add_argument('--stegos', default=True, type=bool)
parser.add_argument('--reveals', default=False, type=bool)
parser.add_argument('--secrets', default=True, type=bool)

params = parser.parse_args()
epoch = params.epoch
make_cover = False
make_stego = params.stegos
make_reveals = params.reveals
make_secrets = params.secrets

cover_dir = '/media/liang/E8B611F5B611C54A/Users/mansh/to_ubuntu/VOC2012/4'
secret_dir = '/media/liang/E8B611F5B611C54A/Users/mansh/to_ubuntu/VOC2012/3'
model_dir = 'models/first_step'
stego_dir = 'result_images/stegos/'
reveal_dir = 'result_images/first_reveals/'

draw_loss = False

batch_size = 1
img_size = 256
pic_need = 1000

# alpha=0.5
# beta=0.3
# gamma=0.85

gpu = params.gpu
gpu_available = True if gpu >= 0 else False

device = torch.device("cuda:%d" % (gpu) if gpu_available else "cpu")

data_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor()
])

dataset_cover = datasets.ImageFolder(cover_dir, data_transform)

dataloader_cover = torch.utils.data.DataLoader(dataset_cover, batch_size=batch_size, shuffle=False, num_workers=1)

dataset_secret = datasets.ImageFolder(secret_dir, data_transform)

dataloader_secret = torch.utils.data.DataLoader(dataset_secret, batch_size=batch_size, shuffle=False, num_workers=1)

# initialize the model and load the params

encoder = model.Encoder()
encoder = encoder.to(device)

# decoder (discriminator)
decoder = model.Decoder()
decoder = decoder.to(device)

ssim_loss = pytorch_ssim.SSIM()
mssim_loss = pytorch_msssim.MSSSIM()
mse_loss = nn.MSELoss()
# dis_loss=nn.BCELoss()


print('loading params')

path = model_dir + '/' + str(epoch) + '.pth.tar'  # load theepoch params

checkpoint = torch.load(path, map_location='cpu')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['deocoder_state_dict'])

cover_ssmi_train = checkpoint['cover_ssmi']
secret_ssmi_train = checkpoint['secret_ssmi']
network_loss = checkpoint['net_loss']
encoder.eval()
decoder.eval()

if draw_loss:
    b = plt.figure()
    plt.plot(cover_ssmi_train)
    plt.title("cover & stego ssmi")
    path = 'result_images/loss_pic/1.png'
    plt.savefig(path)
    c = plt.figure()
    plt.plot(secret_ssmi_train)
    plt.title("secret & reveal ssmi")
    path = 'result_images/loss_pic/2.png'
    plt.savefig(path)
    d = plt.figure()
    plt.plot(network_loss)
    plt.title("network loss")
    path = 'result_images/loss_pic/3.png'
    plt.savefig(path)

    print('finish drawloss')

cover_ssim = []
cover_mse = []
cover_mssim = []
secret_ssim = []
secret_mse = []
secret_mssim = []

stego_step = 0
reveal_step = 0
secret_step = 0
cover_step = 0

for i, data in enumerate(zip(dataloader_cover, dataloader_secret)):

    images = data[0][0]
    ones = data[1][0]

    if len(images) != batch_size: break
    covers = images
    secrets = ones
    #  print(mse_loss(covers,secrets).item())
    secrets = 0 + 0.299 * secrets[:, 0, :, :] + 0.587 * secrets[:, 1, :, :] + 0.114 * secrets[:, 2, :, :]
    secrets = secrets.view(-1, 1, 256, 256)
    # visualize_batch(secrets)
    # print(covers.shape,secrets.shape)
    # transfer it to device
    covers = covers.to(device)
    secrets = secrets.to(device)

    # feed in the network
    # steganalyzer.zero_grad()
    # encoder.zero_grad()
    # decoder.zero_grad()

    stegos = encoder(covers, secrets)

    reveals = decoder(stegos)

    # forward finish

    s_mse = mse_loss(reveals, secrets)
    c_mse = mse_loss(stegos, covers)

    cover_mse.append(c_mse.item())
    secret_mse.append(s_mse.item())

    s_ssim = ssim_loss(reveals, secrets)
    c_ssim = ssim_loss(covers, stegos)

    cover_ssim.append(c_ssim.item())
    secret_ssim.append(s_ssim.item())

    s_mssim = mssim_loss(secrets, reveals)
    c_mssim = mssim_loss(covers, stegos)

    cover_mssim.append(c_mssim.item())
    secret_mssim.append(s_mssim.item())

    if make_cover:
        cover_output = covers.permute(0, 2, 3, 1).cpu().numpy() * 255

        for image in cover_output:
            path = "result_images/covers/" + str(i + 1) + ".png"
            r = np.expand_dims(image[:, :, 0], axis=2)
            g = np.expand_dims(image[:, :, 1], axis=2)
            b = np.expand_dims(image[:, :, 2], axis=2)

            out = np.concatenate((b, g, r), axis=2)
            cv2.imwrite(path, out)

    if make_stego:
        stego_output = stegos.permute(0, 2, 3, 1).detach().cpu().numpy() * 255

        for image in stego_output:
            path = stego_dir + str(i + 1) + ".png"
            r = np.expand_dims(image[:, :, 0], axis=2)
            g = np.expand_dims(image[:, :, 1], axis=2)
            b = np.expand_dims(image[:, :, 2], axis=2)

            out = np.concatenate((b, g, r), axis=2)

            cv2.imwrite(path, out)

    if make_reveals:
        reveals_output = reveals.view(-1, 256, 256, 1).detach().cpu().numpy() * 255

        for image in reveals_output:
            path = reveal_dir + str(i + 1) + ".png"

            cv2.imwrite(path, image)

    if make_secrets:
        onec_secret = secrets.view(-1, 256, 256, 1).detach().cpu().numpy() * 255
        for image in onec_secret:
            path = "result_images/secrets/" + str(i + 1) + ".png"

            cv2.imwrite(path, image)

    if i % 10:
        print(i)

    # if i > pic_need:
    #     break

mean1 = np.mean(cover_ssim)
mean2 = np.mean(secret_ssim)
mean3 = np.mean(cover_mse)
mean4 = np.mean(secret_mse)

print(' ssim: %.4f|%.4f || mse: %.4f|%.4f'
      % (mean1, mean2, mean3, mean4))

print('finish')
