import torch
import torch.nn as nn

class DownBlock(nn.Module):
    """Encoder block with Conv2d + BatchNorm + LeakyReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    """Decoder block with ConvTranspose2d + BatchNorm + ReLU + Dropout"""
    def __init__(self, in_channels, out_channels, dropout=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) if dropout else nn.Identity()
        )
    
    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    """Simplified U-Net Generator for Pix2Pix"""
    def __init__(self, input_channels=3, output_channels=3):
        super().__init__()
        
        # Encoder (downsampling)
        self.enc1 = DownBlock(input_channels, 64)      # 256 -> 128
        self.enc2 = DownBlock(64, 128)                  # 128 -> 64
        self.enc3 = DownBlock(128, 256)                 # 64 -> 32
        self.enc4 = DownBlock(256, 512)                 # 32 -> 16
        self.enc5 = DownBlock(512, 512)                 # 16 -> 8
        self.enc6 = DownBlock(512, 512)                 # 8 -> 4
        self.enc7 = DownBlock(512, 512)                 # 4 -> 2
        
        # Bottleneck (no batchnorm at bottleneck)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Decoder (upsampling) with skip connections
        self.dec1 = UpBlock(512 + 512, 512, dropout=True)   # 2 -> 4
        self.dec2 = UpBlock(512 + 512, 512, dropout=True)   # 4 -> 8
        self.dec3 = UpBlock(512 + 512, 512, dropout=True)   # 8 -> 16
        self.dec4 = UpBlock(512 + 512, 256, dropout=False)  # 16 -> 32
        self.dec5 = UpBlock(256 + 256, 128, dropout=False)  # 32 -> 64
        self.dec6 = UpBlock(128 + 128, 64, dropout=False)   # 64 -> 128
        
        # Final output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        
        # Bottleneck
        b = self.bottleneck(e7)
        
        # Decoder with skip connections (concatenate encoder outputs)
        d1 = self.dec1(torch.cat([b, e7], dim=1))
        d2 = self.dec2(torch.cat([d1, e6], dim=1))
        d3 = self.dec3(torch.cat([d2, e5], dim=1))
        d4 = self.dec4(torch.cat([d3, e4], dim=1))
        d5 = self.dec5(torch.cat([d4, e3], dim=1))
        d6 = self.dec6(torch.cat([d5, e2], dim=1))
        
        # Final output (concatenate with first encoder output)
        output = self.final(torch.cat([d6, e1], dim=1))
        
        return output

class Discriminator(nn.Module):
    """PatchGAN Discriminator - classifies N×N patches"""
    def __init__(self, input_channels=3):
        super().__init__()
        
        # Input is concatenated (distorted + generated/clean) = 6 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Output layer - classifies each patch
        self.output = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
    
    def forward(self, x, y):
        """
        x: distorted image
        y: clean or generated image
        Returns: Patch-wise real/fake predictions
        """
        xy = torch.cat([x, y], dim=1)  # Concatenate along channel dimension
        x = self.block1(xy)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.output(x)

def initialize_weights(model):
    """Initialize weights as per Pix2Pix paper"""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

if __name__ == "__main__":
    # Test the models
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 256, 256)
    
    gen = Generator()
    initialize_weights(gen)
    fake_output = gen(test_input)
    print(f"Generator output shape: {fake_output.shape}")  # Should be [4, 3, 256, 256]
    
    disc = Discriminator()
    initialize_weights(disc)
    disc_output = disc(test_input, fake_output)
    print(f"Discriminator output shape: {disc_output.shape}")  # Should be [4, 1, 30, 30]
    
    print("✓ Model architecture test passed!")