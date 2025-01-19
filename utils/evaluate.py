import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

# Assuming evaluate function is defined as:
def evaluate(model, val_loader, device, amp):
    model.eval()
    criterion = torch.nn.MSELoss()
    ssim_metric = SSIM(data_range=3.1415).to(device)
    psnr_metric = PSNR(data_range=float(3.1415)).to(device)
    total_mse, total_rrmse, total_psnr, total_ssim = 0, 0, 0, 0
    num_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            images, true_masks = batch['image'].to(device), batch['target'].to(device)

            # Predict
            pred_masks = model(images)

            # Calculate MSE loss
            mse_loss = criterion(pred_masks.squeeze(1), true_masks.float())
            total_mse += mse_loss.item()

            # Compute RRMSE
            rrmse = torch.sqrt(mse_loss) / torch.sqrt((true_masks**2).mean())
            total_rrmse += rrmse.item()

            # Compute PSNR
            psnr_value = psnr_metric(pred_masks, true_masks)
            total_psnr += psnr_value.item()

            # Compute SSIM
            ssim_value = ssim_metric(pred_masks, true_masks)
            total_ssim += ssim_value.item()

            num_samples += 1

    # Average over all batches
    avg_mse = total_mse / num_samples
    avg_rrmse = total_rrmse / num_samples
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    return avg_mse, avg_rrmse, avg_psnr, avg_ssim
