import argparse
import logging
import os
import torch

from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.evaluate import evaluate
from utils.data_loading import MatDataset
from models.m_resunet import ResUnetPlusPlus
from models.swim_transformer_v2 import SwinTransformerV2
from models.swim import SCUNet
from models.swim import SCUNet_depth3
# In https://github.com/cszn/DPIR/ drunet is documented as UNetRes:
from models.DRUnet import UNetRes as DRUNet
from models.DRUnet import UNetRes_depth_3 as DRUNet_depth_3
from models.attention_unet import AttentionUNet
from models.unet_other import UNet
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ConstantLR, ExponentialLR, SequentialLR
from torch.optim import RAdam

# Directories
dir_img = 'data/imgs/scaled_dataset_image_by_image.mat'
dir_amp = 'data/imgs/amplitude62000.mat'
dir_checkpoint = Path('./checkpoints/')

def load_model(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs_trained = checkpoint['epoch']
    logging.info(f'Model loaded. Previously trained for {epochs_trained} epochs.')
    return model, optimizer, epochs_trained

def train_model(model, device, epochs=5, start_epoch=0, batch_size=1, global_step=0, learning_rate=1e-4, val_percent=0.1, save_checkpoint=True, img_scale=1, amp=False):
    
    # Create dataset
    full_dataset = MatDataset(dir_img, dir_amp, img_scale)
    # First, create a subset of the full dataset
    subset_indices = torch.randperm(len(full_dataset))[:int(len(full_dataset) * 0.01)]

    dataset = torch.utils.data.Subset(full_dataset, subset_indices) 

    # First, split off the test set
    train_val_dataset, test_set = train_test_split(
        dataset, 
        test_size=0.1,  # 10% for test set
        random_state=42  # fixed random seed for reproducibility
    )
    # Then split the remaining data into train and validation sets
    train_set, val_set = train_test_split(
        train_val_dataset, 
        test_size=val_percent / (1 - 0.1),  # adjust validation percentage 
        random_state=42  # same seed to ensure reproducibility
    )

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_args)

    # Initialize logging
    logging.info(f'Starting training: Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}, Training size: {len(train_set)}, Validation size: {len(val_set)}')

    # Set up optimizer and loss
    optimizer = RAdam(model.parameters(), lr=learning_rate)
    gamma = 0.9  # Adjust gamma as needed; this is the decay factor
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    criterion = torch.nn.MSELoss()
    global_step = global_step

    # Begin training
    for epoch in range((start_epoch + 1), epochs + 1):

        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_set), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'].to(device), batch['target'].to(device)  # Move to device

                # Forward pass
                masks_pred = model(images)
                loss = criterion(masks_pred.squeeze(1), true_masks.float())

                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

            scheduler.step()

        avg_mse, avg_rrmse, avg_psnr, avg_ssim = evaluate(model, val_loader, device, amp)
        logging.info(f'Validation MSE: {avg_mse}, RRMSE: {avg_rrmse}, PSNR: {avg_psnr}, SSIM: {avg_ssim}')

        if save_checkpoint:
            checkpoint_path = dir_checkpoint / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / len(train_loader),
            }, checkpoint_path) 
            logging.info(f'Checkpoint saved at {checkpoint_path}')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', type=int, default=80, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', type=float, default=10.0, help='Validation percent (0-100)')          
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    #model = UNet(num_classes=1, in_channels=1, depth=5).to(device)
    #model = UNet(num_classes=1, in_channels=1, depth=4).to(device)

    
    #model = DRUNet(in_nc=1, nb=4).to(device)
    model = DRUNet_depth_3(in_nc=1, nb=4).to(device)

    # config is scunet is the number of Swin-Conv (SC) Blocks per layer
    #model = SCUNet(input_resolution=128, in_nc=1, config=[4,4,4,4,4,4,4], drop_path_rate=0.1).to(device=device)
    #model = SCUNet_depth3(input_resolution=128, in_nc=1, config=[4,4,4,4,4,4,4], drop_path_rate=0.1).to(device=device)
    
    # Experimental:
        #model = ResUnetPlusPlus(channel=1).to(device)
        #model = SCUNet(input_resolution=128, in_nc=1, config=[4,4,4,4,4,4,4], drop_path_rate=0.1).to(device=device)
        #NOTWORKING model = SwinTransformerV2(img_size=128, in_chans=1, num_classes=0, window_size=8).to(device=device)
        #model = AttentionUNet(img_ch=1, output_ch=1).to(device)
        #model = Hiera((128,128), num_classes=1, in_chans=1, stages=(2, 2, 6, 2)).to(device)
        #model = Rec_Transformer(input_size=128, in_chans=1).to(device)
        #model = UNet_from_drunet_lib(in_nc=1, out_nc=1).to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    # Check if the model has already been trained
    checkpoint_path = dir_checkpoint / 'checkpoint_epoch_1.pth'
    
    if os.path.exists(checkpoint_path):
        logging.info('Loading saved model...')
        model, optimizer, epochs = load_model(model, optimizer, checkpoint_path)
        train_model(model=model, device=device, epochs=args.epochs, start_epoch=epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, img_scale=args.scale, val_percent=args.validation / 100, amp=args.amp)
    else:
        # Train model if no checkpoint exists
        try:
            train_model(model=model, device=device, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, img_scale=args.scale, val_percent=args.validation / 100, amp=args.amp)
        except torch.cuda.OutOfMemoryError:
            logging.error('OutOfMemoryError! Consider reducing batch size or using mixed precision.')
