from __future__ import print_function
# %matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# 为再现性设置随机seem
manualSeed = 999
# manualSeed = random.randint(1, 10000) # 如果你想要新的结果就是要这段代码
# print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# 数据集根目录
dataroot = "data/celeba"
# 加载数据的工作线程数
workers = 2
# 训练期间的batch大小
batch_size = 128
# 训练图像的空间大小。所有图像将调整为此大小
image_size = 64
# 训练图像中的通道数
nc = 3
# 输入向量大小
nz = 100
# 生成器中特征图大小
ngf = 64
# 判别器中特征映射大小
ndf = 64
num_epochs = 5
# 优化器的学习率
lr = 0.0002
# 使用于Adam优化器Beta1的超参数
beta1 = 0.5
# 可用的GPU数量0-cpu
ngpu = 1

# 创建数据集
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# 创建加载器
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
# 选择我们运行的设备
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def drawTest():
    # 绘制部分输入图像测试
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()


# 权重初始化，所有模型权重从正态分布中随机初始化
# weights_init函数将初始化模型作为 输入，并重新初始化所有卷积，
# 卷积转置和batch标准化层以满足此标准。初始化后立即将此函数应用于模型。
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# 生成器，将矢量映射到数据空间（这里是图像）最终是创建3*64*64的图像
# 生成器代码
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()  # 对父类属性初始化
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入是Z，进入卷积
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# ------------------------
# 创建生成器
netG = Generator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1): netG = nn.DataParallel(netG, list(range(ngpu)))
# 用weight_init随机初始化所有权重
netG.apply((weights_init))
#print(netG)
# -----------------------


# 判别器是二元分类网络，将图像作为输入，输出图像是真实的标量概率
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# ----------------------------------------
# 创建判别器
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# 应用weights_init函数随机初始化所有权重，mean= 0，stdev = 0.2
netD.apply(weights_init)

# 打印模型
#print(netD)
# ----------------------------------------

# 损失函数和优化器
# 初始化BCELoss函数
criterion = nn.BCELoss()

# 创建一批潜在的向量，我们将用它来可视化生成器的进程
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# 在训练期间建立真假标签的惯例
real_label = 1
fake_label = 0

# 为 G 和 D 设置 Adam 优化器
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# 训练
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []


def train():
    z_sample=torch.randn(100, nz, 1, 1, device=device)
    print("Starting Training Loop...")
    iters = 0
    # For each epoch
    for epoch in range(num_epochs):
        # 对于数据加载器中的每个batch
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device, dtype=torch.float)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    # filename="cele_train/Epoch"+str(epoch)+".jpg"
                    # torchvision.utils.save_image(fake,filename,nrow=8)
                    # #画图
                    # grid_img = torchvision.utils.make_grid(fake.cpu(), nrow=8)
                    # plt.figure(figsize=(8, 8))
                    # plt.imshow(grid_img.permute(1, 2, 0))
                    # plt.show()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1





# 损失与训练迭代关系图
def plot_loss():
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

#生成训练过程中的图片
def show_img_list():
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())

# 用模型生成图片
def get_new_img():
    generator = Generator(ngpu).to(device)
    generator.load_state_dict(torch.load('dcgan_g.pth'))
    noise = torch.randn(1, nz, 1, 1, device=device)
    fake = generator(noise).detach().cpu()
    filename = "cele_train/test.jpg"
    torchvision.utils.save_image(fake, filename)




if __name__ == '__main__':
    train()
    torch.save(netG.state_dict(), 'dcgan_g.pth')
    torch.save(netD.state_dict(), 'dcgan_d.pth')
    plot_loss()
