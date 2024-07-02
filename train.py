import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

import mobileclip
from RATLIP import NetG, NetC, NetD
from datasets import ImageSet

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Model Params
imsize = 256
zdim = 100
cdim = 512

# Training Params
epochs = 100
loadpt = 1

clip, _, preprocess = mobileclip.create_model_and_transforms(f'mobileclip_s0', pretrained=f'./models/mobileclip_s0.pt')
clip = clip.to(device, dtype=dtype)

for p in clip.image_encoder.parameters():
    p.requires_grad = False
clip.image_encoder.eval()

for p in clip.text_encoder.parameters():
    p.requires_grad = False
clip.text_encoder.eval()

dataset = ImageSet("C:/Datasets/MSCOCO/train2017", preprocess)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

netG = NetG(64, zdim, cdim, imsize, clip).to(device, dtype=dtype)
netD = NetD().to(device, dtype=dtype)
netC = NetC(cdim).to(device, dtype=dtype)

if loadpt > -1:
    netG.load_state_dict(torch.load(f'./models/netG_{loadpt}.pth').state_dict())
    netD.load_state_dict(torch.load(f'./models/netD_{loadpt}.pth').state_dict())
    netC.load_state_dict(torch.load(f'./models/netC_{loadpt}.pth').state_dict())

print("G:", np.sum([p.numel() for p in netG.parameters()]).item()/10**6)
print("D:", np.sum([p.numel() for p in netD.parameters()]).item()/10**6)
print("C:", np.sum([p.numel() for p in netC.parameters()]).item()/10**6)

optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
optimizerD = optim.Adam(list(netD.parameters()) + list(netC.parameters()), lr=0.0004, betas=(0.0, 0.9))
scalerG = torch.cuda.amp.GradScaler()
scalerD = torch.cuda.amp.GradScaler()

tnoise = torch.randn(1, zdim).to(device)
tembed = clip.encode_image(preprocess(Image.open("./dogcat.jpg").convert('RGB')).cuda().unsqueeze(0))

def MAGP(img, sent, out, scaler):
    grads = torch.autograd.grad(outputs=scaler.scale(out),
                            inputs=(img, sent),
                            grad_outputs=torch.ones_like(out),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    inv_scale = 1./(scaler.get_scale()+float("1e-8"))
    grads = [grad * inv_scale for grad in grads]
    with torch.cuda.amp.autocast():
        grad0 = grads[0].view(grads[0].size(0), -1)
        grad1 = grads[1].view(grads[1].size(0), -1)
        grad = torch.cat((grad0,grad1),dim=1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp =  2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp

for epoch in range(epochs):
    for i, images in enumerate(dataloader):

        batch_size = images.size(0)
        images = images.to(device, dtype=dtype)
        
        # Train D
        optimizerD.zero_grad()
        with torch.cuda.amp.autocast():
            rfeats, rembeds = clip.encode_image(images, features=True)
            images.requires_grad = True
            rembeds.requires_grad = True

            rexfts, pfeats = netD(rfeats)
            closs = netC(rexfts, rembeds)
            rloss = torch.mean(F.relu(1. - closs))

            membeds = torch.cat((rembeds[1:], rembeds[0:1]), dim=0).detach()
            mloss = torch.mean(F.relu(1. + netC(rexfts, membeds)))

            noise = torch.randn(batch_size, zdim).to(device)
            fake = netG(noise, rembeds)
            ffeats, fembeds = clip.encode_image(fake, features=True)
            fexfts, _ = netD(ffeats)
            floss = torch.mean(F.relu(1. + netC(fexfts, rembeds)))

        ploss = MAGP(pfeats, rembeds, closs, scalerD)
        with torch.cuda.amp.autocast():
            dloss = rloss + (floss + mloss) / 2.0 + ploss
        scalerD.scale(dloss).backward(retain_graph=True)
        scalerD.step(optimizerD)
        scalerD.update()
        if scalerD.get_scale() < 64:
            scalerD.update(16384.0)

        # for l in [mloss, rloss, floss, fexfts, fembeds, fake]:
        #     print(f'({l.min().item()}, {l.max().item()})', end=' ')
        # print()

        # Train G
        optimizerG.zero_grad()
        with torch.cuda.amp.autocast():
            fexfts, _ = netD(ffeats)
            closs = netC(fexfts, rembeds)
            gloss = -closs.mean() - 4.0 * torch.cosine_similarity(fembeds, rembeds).mean()
        scalerG.scale(gloss).backward()
        scalerG.step(optimizerG)
        scalerG.update()
        if scalerG.get_scale() < 64:
            scalerG.update(16384.0)
        
        if i % 50 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {dloss.item()}] [G loss: {gloss.item()}] [CLIP: {torch.cosine_similarity(fembeds, rembeds).mean().item()}]")
            transforms.ToPILImage()(netG(tnoise, tembed)[0]).save(f"./results/{epoch}-{i}.png")
            torch.save(netG, f"./models/netG_{epoch}.pth")
            torch.save(netD, f"./models/netD_{epoch}.pth")
            torch.save(netC, f"./models/netC_{epoch}.pth")
