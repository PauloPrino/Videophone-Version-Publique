from pillow_heif import register_heif_opener
from PIL import Image
import os

def convert_heic_to_jpg(heic_path, jpg_path):
    # Register the HEIF opener with Pillow
    register_heif_opener()

    # Open the HEIC file
    img = Image.open(heic_path)

    # Convert and save as JPG
    img.convert("RGB").save(jpg_path, "JPEG")

def convert_all_heic_in_folder(folder):
    # Ensure the folder exists
    if not os.path.isdir(folder):
        raise ValueError(f"The folder {folder} does not exist.")

    # List all files in the folder
    files = os.listdir(folder)

    # Convert each HEIC file to JPG
    for file in files:
        if file.endswith('.heic'):
            heic_path = os.path.join(folder, file)
            jpg_path = os.path.join(folder, file.replace('.heic', '.jpg'))
            convert_heic_to_jpg(heic_path, jpg_path)
    print(f"Converted all HEIC files in {folder} to JPG.")

def delete_heic_files(folder):
    # Ensure the folder exists
    if not os.path.isdir(folder):
        raise ValueError(f"The folder {folder} does not exist.")

    # List all files in the folder
    files = os.listdir(folder)

    # Delete each HEIC file
    for file in files:
        if file.endswith('.heic'):
            heic_path = os.path.join(folder, file)
            os.remove(heic_path)
    print(f"Deleted all HEIC files in {folder}.")

convert_all_heic_in_folder("Labeled_data/AmazonPhotos (3)")
delete_heic_files("Labeled_data/AmazonPhotos (3)")