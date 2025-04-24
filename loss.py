import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Combined loss function using MSE and SSIM.
        
        Args:
            alpha (float): Weight for MSE loss. SSIM loss weight will be (1-alpha).
                          Default is 0.5 for equal weighting.
        """
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        
    def forward(self, pred, target):
        """
        Calculate the combined loss.
        
        Args:
            pred (torch.Tensor): Predicted (denoised) image
            target (torch.Tensor): Target (original) image
            
        Returns:
            torch.Tensor: Combined loss value
        """
        mse_loss = self.mse(pred, target)
        ssim_loss = 1 - self.ssim(pred, target)  # Convert SSIM to loss (1 - SSIM)
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss 
