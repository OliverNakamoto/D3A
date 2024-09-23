import imagehash
from PIL import Image
import os


def count_jpg_files(folder):
    count = 0
    for filename in os.listdir(folder):
        if filename.lower().endswith('.jpg'):
            count += 1
    return count

# Specify your folder name
folder_name = 'final_dataset'


def deduplicate_images(image_folder):
    hashes = {}
    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        path = os.path.join(image_folder, filename)
        try:
            img = Image.open(path)
            hash = imagehash.phash(img)
            if hash in hashes:
                os.remove(path)
                print(f"Removed duplicate image: {filename}")
            else:
                hashes[hash] = filename
        except Exception as e:
            print(f"Error processing {path}: {e}")


jpg_count = count_jpg_files(folder_name)

# Deduplicate final dataset
deduplicate_images('final_dataset')


jpg_count2 = count_jpg_files(folder_name)

print(jpg_count, jpg_count2)