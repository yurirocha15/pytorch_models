from torch import nn
from torch import cat

def init_weights(model):
    layer_name = model.__class__.__name__
    if layer_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif layer_name.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

class UNetCell(nn.Module):
    '''
    Outer UNet cell definition:
    input     ---->         Encoder ----> inner_module ----> Decoder     ---->      output

    General UNet cell definition:
    input  ---->    |                                                       A ----> output
                    |downsample                                     upsample|
                    V ----> Encoder ----> inner_module ----> Decoder ---->  |

    Central UNet cell definition:
    input  ---->    |                                                       A ----> output
                    |downsample                                     upsample|
                    V       ---->       Center Convolutions     ---->       |
    '''
    def __init__(self, in_filters, n_kernels, use_batch_norm=True, inner_module=None, is_outer_layer=False):
        super(UNetCell, self).__init__()
        self.is_outer_layer = is_outer_layer
        #DownSampling
        downsampling = [nn.AvgPool2d(kernel_size=2, stride=2)]

        #Center Module
        center = [nn.Conv2d(in_filters, n_kernels, kernel_size=3, stride=1, padding=1, bias=False)]
        if use_batch_norm:
            center.append(nn.BatchNorm2d(n_kernels))
        center.extend([
            nn.LeakyReLU(0.1),
            nn.Conv2d(n_kernels, n_kernels, kernel_size=3, stride=1, padding=1, bias=False)
        ])
        if use_batch_norm:
            center.append(nn.BatchNorm2d(n_kernels))
        center.append(nn.LeakyReLU(0.1))

        #Encoder
        encoder = [nn.Conv2d(in_filters, n_kernels, kernel_size=3, stride=1, padding=1, bias=False)]
        if use_batch_norm:
            encoder.append(nn.BatchNorm2d(n_kernels))
        encoder.extend([
            nn.LeakyReLU(0.1),
            nn.Conv2d(n_kernels, n_kernels, kernel_size=3, stride=1, padding=1, bias=False)
        ])
        if use_batch_norm:
            encoder.append(nn.BatchNorm2d(n_kernels))
        encoder.append(nn.LeakyReLU(0.1))

        #Decoder
        decoder = [nn.Conv2d(n_kernels*2, n_kernels, kernel_size=3, stride=1, padding=1, bias=False)]
        if use_batch_norm:
            decoder.append(nn.BatchNorm2d(n_kernels))
        decoder.extend([
            nn.LeakyReLU(0.1),
            nn.Conv2d(n_kernels, n_kernels, kernel_size=3, stride=1, padding=1, bias=False)
        ])
        if use_batch_norm:
            decoder.append(nn.BatchNorm2d(n_kernels))
        decoder.append(nn.LeakyReLU(0.1))

        #UpSampling
        upsampling = [nn.ConvTranspose2d(n_kernels, in_filters, kernel_size=4, stride=2, padding=1, bias=False)]
        if use_batch_norm:
            upsampling.append(nn.BatchNorm2d(in_filters))
        upsampling.append(nn.ReLU())

        if inner_module is None:
            self.main = nn.Sequential(
                *downsampling,
                *center,
                *upsampling
            )
        elif is_outer_layer:
            self.main = nn.Sequential(
                *encoder,
                inner_module, # inner module is already a Pytorch Module
                *decoder
            )
        else:
            self.main = nn.Sequential(
                *downsampling,
                *encoder,
                inner_module, # inner module is already a Pytorch Module
                *decoder,
                *upsampling
            )

    def forward(self, X):
        if self.is_outer_layer:
            return self.main(X)
        else:
            return cat([X, self.main(X)], 1)

class UNet512(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_kernels=32, use_batch_norm=True):
        super(UNet512, self).__init__()

        unet_cell = UNetCell(in_filters=n_kernels*16, n_kernels=n_kernels*32, use_batch_norm=use_batch_norm)
        unet_cell = UNetCell(in_filters=n_kernels*8, n_kernels=n_kernels*16, use_batch_norm=use_batch_norm, inner_module=unet_cell)
        unet_cell = UNetCell(in_filters=n_kernels*4, n_kernels=n_kernels*8, use_batch_norm=use_batch_norm, inner_module=unet_cell)
        unet_cell = UNetCell(in_filters=n_kernels*2, n_kernels=n_kernels*4, use_batch_norm=use_batch_norm, inner_module=unet_cell)
        unet_cell = UNetCell(in_filters=n_kernels, n_kernels=n_kernels*2, use_batch_norm=use_batch_norm, inner_module=unet_cell)
        unet_cell = UNetCell(in_channels, n_kernels=n_kernels, use_batch_norm=use_batch_norm, inner_module=unet_cell, is_outer_layer=True)

        self.main = nn.Sequential(
            unet_cell,
            nn.Conv2d(n_kernels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

        self.apply(init_weights)

    def forward(self, X):
        return self.main(X)

    def print(self):
        print(self)
        

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6, use_batch_norm=True):
        super(PatchDiscriminator, self).__init__()
        self.main = [nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1)]
        if use_batch_norm:
            self.main.append(nn.BatchNorm2d(64))
        self.main.extend([
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        ])
        if use_batch_norm:
            self.main.append(nn.BatchNorm2d(128))
        self.main.extend([
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        ])
        if use_batch_norm:
            self.main.append(nn.BatchNorm2d(256))
        self.main.extend([
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1)
        ])
        if use_batch_norm:
            self.main.append(nn.BatchNorm2d(512))
        self.main.extend([
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        ])

        self.main = nn.Sequential(*self.main)

        self.apply(init_weights)

    def forward(self, X):
        return self.main(X)

    def print(self):
        print(self)


