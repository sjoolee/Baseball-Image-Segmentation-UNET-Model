import torch
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm #progress_bar
import torch.nn as nn
import torch.optim as optim 
from model import UNET
from utils import (
    save_checkpoint,
    load_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

#Hyperparameters & Configuration.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 #may need to decrease
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 160 #originally 1280 height and 1918 for width
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "baseball_data/train/"
TRAIN_MASK_DIR = "baseball_data/train_masks/"
VAL_IMG_DIR = "baseball_data/valid/"
VAL_MASK_DIR = "baseball_data/valid_masks/"


def train_fn(loader, model, optimizer, loss_fn, scaler): #performs the data augmentation for training and validation sets using albumentations library, intilizes the U-Net model, loss function (Binary Cross Entropy w Logits) and optimizer (Adam)
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE) 

        #forward pass
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        #backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm progress loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose (
        [
            A.Resize(height=IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean = [0.0,0.0,0.0],
                std = [1.0,1.0,1.0],
                max_pixel_value=255.0, #may not need 
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() #for more segmentation it would be out_channel = # of classes and loss function = cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model) #if the model is loaded, flag is set to True and it proceeds to load a pre-trained model checkpoint


    check_accuracy(val_loader, model, device=DEVICE) #computes and prints accuracy of the model on the validation set
    scaler = torch.cuda.amp.GradScaler() #initializes gradient scaler object for mixed-precision training #used to scale the loss value during backpropagation to prevent underflow/overflow of gradients when using float16 data types

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder #saves predictions made by the model on the validation set images
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )


if __name__ == "__main__":
    main()
    