import os
from PIL import Image

# Assuming your dataset directory structure is organized such that images and masks are paired together
image_dir = "C:/Users/rainlab/Downloads/baseball_data/valid"
mask_dir = "C:/Users/rainlab/Downloads/baseball_data/valid_masks"

# Get the list of image files
image_files = os.listdir(image_dir)

# Iterate through each image
for image_file in image_files:
    # Construct paths for image and corresponding mask
    image_path = os.path.join(image_dir, image_file)
    mask_file = image_file.replace(".jpg", "_mask.png")  # Assuming mask files have a similar naming convention
    mask_path = os.path.join(mask_dir, mask_file)
    
    # Open the images
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    
    # Get dimensions of image and mask
    image_width, image_height = image.size
    mask_width, mask_height = mask.size
    
    # Compare dimensions
    if image_width != mask_width or image_height != mask_height:
        print(f"Image {image_file} and its corresponding mask have different dimensions.")
