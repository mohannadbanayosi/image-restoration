import torch.nn as nn
import torch


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
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(0.2)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(0.2)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(0.2)
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(0.2)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(0.2)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # TODO: experiment with `nn.Tanh`
        )

    def forward(self, x):
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        
        dec3_out = self.dec3(enc3_out)
        dec3_out = torch.cat([dec3_out, enc2_out], dim=1)
        
        dec2_out = self.dec2(dec3_out)
        dec2_out = torch.cat([dec2_out, enc1_out], dim=1)
        
        out = self.dec1(dec2_out)
        return out
