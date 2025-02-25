import os
import tomllib
from pathlib import Path

from modules.model.classifier import ResNetASPPClassifier
from modules.preprocessing.image_cropper import ImageCropper

# Preprocessing pipeline
def preprocess(config):
    # If there is an index file already at the output and cached preprocessing is on,
    # skip this step
    if config["UsePreprocessing"] and config["UseCachedPreprocessing"]:
        index_filepath = Path(config["OutputRoot"]) / "index.csv"
        index_filepath = index_filepath.as_posix()
        if os.path.isfile(index_filepath):
            print(f"Using cached image crop index file at {index_filepath}, image cropping skipped")
            return

    if config["UsePreprocessing"]:
        cropper = ImageCropper(
            config["OutputRoot"],
            config["BaseInputDir"],
            dirs=config["Dirs"],
            recurse=config["InputRecurse"]
        )
        cropper.begin_cropping()

def train(config):
    if config["DoTraining"]:
        classifier = ResNetASPPClassifier(config)
        classifier.load_data()

        # Fix: Ensure labels are converted to LongTensor during training
        for batch_idx, (images, labels) in enumerate(classifier.train_loader):
            images = images.float().to(classifier.device)
            # Convert labels to LongTensor
            labels = labels.to(classifier.device).long()

            if batch_idx == 0:
                # Debug: Check label dtypes
                print(f"Label dtype: {labels.dtype}")

        # Loading cached model is for eval purposes, not for retraining
        if not config["UseCachedModel"]:
            classifier.train()

def full_train(config):
    preprocess(config["preprocessing"])
    train(config["training"])

    print("Done")

def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    full_train(config)

if __name__ == "__main__":
    main()