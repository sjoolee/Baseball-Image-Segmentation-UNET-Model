'''
organizing the images to be sorted into 2 folders within the test, train and valid folder => image and mask

There were frame# in the original dataset so we isolated for File '6pBQVjveD6o171_mp4_frame58_jpg.rf.c24b4e8c81c07583e2f652c58d9c73a5_mask.png' to be simplified to 'frame58.png' 
Images that had the same frame# were labelled #+1 for each time there was a same frame# value 

'''


import os
import shutil
import re

def organize_images(folder_path, new_prefix):
    try:
        # List all files in the folder
        files = os.listdir(folder_path)
        image_count = 1
        mask_count = 1
        
        # Create folders if not exist
        image_folder = os.path.join(folder_path, 'image')
        mask_folder = os.path.join(folder_path, 'mask')
        os.makedirs(image_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)
        
        jpg_files = [file for file in files if file.lower().endswith('.jpg')]
        png_files = [file for file in files if file.lower().endswith('.png')]

        # Organize images into folders
        for idx, filename in enumerate(jpg_files):
            dest_folder = image_folder
            
            # Construct new name
            match = re.search(r'frame\d+', filename)
            simplified_name = match.group() if match else f'image{image_count}'
            simplified_name += '.jpg'
            
            # Move the file to the destination folder
            shutil.move(os.path.join(folder_path, filename), os.path.join(dest_folder, simplified_name))
            print(f"File '{filename}' moved to '{dest_folder}' successfully.")

        for idx, filename in enumerate(png_files):
            dest_folder = mask_folder
            
            # Construct new name
            match = re.search(r'frame\d+', filename)
            simplified_name = match.group() if match else f'image{mask_count}'
            simplified_name += '_mask.png'
            
            # Move the file to the destination folder
            shutil.move(os.path.join(folder_path, filename), os.path.join(dest_folder, simplified_name))
            print(f"File '{filename}' moved to '{dest_folder}' successfully.")
    
    except FileNotFoundError:
        print(f"Folder '{folder_path}' not found.")

# Example usage:
folder_path = "C:/Users/rainlab/Downloads/baseball_data/test"
new_prefix = "image"

organize_images(folder_path, new_prefix)
