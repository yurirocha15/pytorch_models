from pix2pix import networks
from tqdm import tqdm
from torch import tensor
from torch import nn
from torch import cat
from torch import no_grad
from torch import cuda
from torch.optim import Adam
from torchvision.utils import make_grid
from torchvision.utils import save_image

class Pix2Pix(nn.Module):
    def __init__(self, dataloader, test_dataloader, lr_g=0.0002, lr_d=0.0001, in_channels=3, out_channels=3, batch_size=32, lambdaL1=100.0, isTrain=True):
        super(Pix2Pix, self).__init__()

        if cuda.is_available():
            cuda.set_device(0)

        self.batch_size = batch_size
        self.lambdaL1 = lambdaL1
        self.isTrain = isTrain
        self.register_buffer('real_label', tensor(1.0))
        self.register_buffer('fake_label', tensor(0.0))

        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        data = enumerate(self.test_dataloader)
        _, data = next(data)
        batch_size = len(data['A'])
        batch_size = batch_size if batch_size < 8 else 8
        self.test_images = data['A'][0:batch_size]
        if cuda.is_available():
            self.test_images = self.test_images.cuda()

        self.modelG = networks.UNet512(in_channels, out_channels)
        if cuda.is_available():
            self.modelG.cuda()

        if self.isTrain:
            self.modelD = networks.PatchDiscriminator(in_channels + out_channels)
            if cuda.is_available():
                self.modelD.cuda()

            self.criterion = nn.BCELoss()
            self.criterionL1 = nn.L1Loss()

            self.optimizerG = Adam(self.modelG.parameters(), lr=lr_g, betas=(0.5, 0.999))
            self.optimizerD = Adam(self.modelD.parameters(), lr=lr_d, betas=(0.5, 0.999))


    def forward(self, X):
        return self.modelG(X)

    def train(self, epochs):
        if cuda.is_available():
            self.modelG.cuda()
            self.modelD.cuda()

        self.losses_G = []
        self.losses_D = []
        self.img_list = []

        count = 0
        for epoch in tqdm(range(epochs), desc='Epochs'):
            for i, data in enumerate(tqdm(self.dataloader, desc='Batches'), 0):
                input_imgs, real_imgs = data['A'], data['B']
                if cuda.is_available():
                    input_imgs, real_imgs = input_imgs.cuda(), real_imgs.cuda()

                ################
                # Discriminator
                ################
                
                self.set_requires_grad(self.modelD, True)
                self.optimizerD.zero_grad()

                real_input = cat((input_imgs, real_imgs), 1)
                pred_real = self.modelD(real_input)
                target_real = self.get_target(pred_real, is_real=True)
                if cuda.is_available():
                    target_real = target_real.cuda()
                loss_D_real = self.criterion(pred_real, target_real)

                fake_imgs = self.forward(input_imgs)

                fake_input = cat((input_imgs, fake_imgs), 1)
                pred_fake = self.modelD(fake_input.detach())
                target_fake = self.get_target(pred_fake, is_real=False)
                if cuda.is_available():
                    target_fake = target_fake.cuda()
                loss_D_fake = self.criterion(pred_fake, target_fake)

                loss_D = (loss_D_real + loss_D_fake)/2.0

                loss_D.backward()
                self.optimizerD.step()

                ################
                # Generator
                ################
                self.set_requires_grad(self.modelD, False)
                self.optimizerG.zero_grad()

                fake_input = cat((input_imgs, fake_imgs), 1)
                pred_fake2 = self.modelD(fake_input)
                target_fake = self.get_target(pred_fake2, is_real=True)
                if cuda.is_available():
                    target_fake = target_fake.cuda()
                loss_G_pred = self.criterion(pred_fake2, target_fake)
                loss_G_L1 = self.criterionL1(fake_imgs, real_imgs) * self.lambdaL1
                loss_G = loss_G_pred + loss_G_L1
                loss_G.backward()
                self.optimizerG.step()

                if i % 5 == 0:
                    tqdm.write(f'''[{epoch}/{epochs}][{i}/{len(self.dataloader)}] \
                         LossD: {loss_D.item():.4f} (real: {loss_D_real:.4f}, fake: {loss_D_fake:.4f}) \
                         LossG: {loss_G.item():.4f} (pred: {loss_G_pred:.4f}, L1: {loss_G_L1:.4f}) \
                         D(x): {pred_real.mean().item():.4f} \
                         D(G(z)): {pred_fake.mean().item():.4f} -> {pred_fake2.mean().item():.4f}''')
                
                self.losses_G.append(loss_G.item())
                self.losses_D.append(loss_D.item())


                if count % 100 == 0 or (epoch == epochs - 1 and i == len(self.dataloader) - 1):
                    with no_grad():
                        fakes = self.modelG(self.test_images).detach().cpu()
                        grid = make_grid(fakes, padding=2, normalize=True)
                    self.img_list.append(grid)
                    save_image(grid, f"./visualization/{count}.png", normalize=False)

                count += 1





    def get_target(self, pred_tensor, is_real):
        target = self.real_label if is_real else self.fake_label
        return target.expand_as(pred_tensor)

    def set_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad
