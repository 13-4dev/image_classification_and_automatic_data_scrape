import os
import shutil
from model_train import train_and_save_model
from scrape import download_images
from load_model import load_and_predict_image

def setup_dataset(object1, object2):
    # data dir
    base_dir = r"binary-classification-objects\data"
    object1_dir = os.path.join(base_dir, "object1")
    object2_dir = os.path.join(base_dir, "object2")
    
    # clear data
    if os.path.exists(object1_dir):
        shutil.rmtree(object1_dir)
    if os.path.exists(object2_dir):
        shutil.rmtree(object2_dir)

    os.makedirs(object1_dir)
    os.makedirs(object2_dir)

    print(f"Downloading {object1}...")
    download_images(object1, object1_dir)
    print(f"Downloading {object2}...")
    download_images(object2, object2_dir)

    print(f"Contents of {object1_dir}: {os.listdir(object1_dir)}")
    print(f"Contents of {object2_dir}: {os.listdir(object2_dir)}")


if __name__ == "__main__":
    object1 = input("Enter the name of the first object (for example, car): ")
    object2 = input("Enter the name of the second object (for example, truck): ")
    dataset_dir = r"binary-classification-objects\data"

    setup_dataset(object1, object2)
    
    train_and_save_model(dataset_dir)

    load_and_predict_image()
