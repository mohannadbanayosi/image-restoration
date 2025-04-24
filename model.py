import torch.nn as nn
import torch

class DenoisingAutoencoderMini(nn.Module):

    def get_model_architecture(self):
        """Extract model architecture information."""
        architecture_info = {
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "layers": []
        }
        
        # Extract information about each layer
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                layer_info = {
                    "name": name,
                    "type": module.__class__.__name__,
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": module.kernel_size,
                    "stride": module.stride,
                    "padding": module.padding
                }
                architecture_info["layers"].append(layer_info)
        
        return architecture_info

    def __init__(self):
        super(DenoisingAutoencoderMini, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(0.2)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(0.2)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(0.2)
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(0.2)
        )

        self.enc5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(0.2)
        )

        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(0.2)
        )
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(0.2)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(0.2)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(0.2)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # TODO: experiment with `nn.Tanh`
        )

    def forward(self, x):
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        enc5_out = self.enc5(enc4_out)
        
        dec5_out = self.dec5(enc5_out)
        dec5_out = dec5_out + enc4_out
        
        dec4_out = self.dec4(dec5_out)
        dec4_out = dec4_out + enc3_out
        
        dec3_out = self.dec3(dec4_out)
        dec3_out = dec3_out + enc2_out
        
        dec2_out = self.dec2(dec3_out)
        dec2_out = dec2_out + enc1_out
        
        out = self.dec1(dec2_out)
        return out


class DenoisingAutoencoder(nn.Module):

    def get_model_architecture(self):
        """Extract model architecture information."""
        architecture_info = {
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "layers": []
        }
        
        # Extract information about each layer
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                layer_info = {
                    "name": name,
                    "type": module.__class__.__name__,
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": module.kernel_size,
                    "stride": module.stride,
                    "padding": module.padding
                }
                architecture_info["layers"].append(layer_info)
        
        return architecture_info

    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(0.2)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(0.2)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(0.2)
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(0.2)
        )

        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(0.2)
        )

        self.enc6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(0.2)
        )

        self.enc7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(0.2)
        )

        self.dec7 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(0.2)
        )
        
        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(0.2)
        )
        
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(0.2)
        )
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(0.2)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(0.2)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(0.2)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # TODO: experiment with `nn.Tanh`
        )

    def forward(self, x):
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        enc5_out = self.enc5(enc4_out)
        enc6_out = self.enc6(enc5_out)
        enc7_out = self.enc7(enc6_out)

        dec7_out = self.dec7(enc7_out)
        dec7_out = dec7_out + enc6_out
        
        dec6_out = self.dec6(dec7_out)
        dec6_out = dec6_out + enc5_out
        
        dec5_out = self.dec5(dec6_out)
        dec5_out = dec5_out + enc4_out
        
        dec4_out = self.dec4(dec5_out)
        dec4_out = dec4_out + enc3_out
        
        dec3_out = self.dec3(dec4_out)
        dec3_out = dec3_out + enc2_out
        
        dec2_out = self.dec2(dec3_out)
        dec2_out = dec2_out + enc1_out
        
        out = self.dec1(dec2_out)
        return out
