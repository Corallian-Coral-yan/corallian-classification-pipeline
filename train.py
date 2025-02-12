import os
import tomllib

from modules.model.classifier import ResNetClassifier
from modules.preprocessing.image_cropper import ImageCropper

# Preprocessing pipeline
def preprocess(config):
    # If there is an index file already at the output and cached preprocessing is on,
    # skip this step
    if config["UsePreprocessing"] and config["UseCachedPreprocessing"]:
        index_filepath = os.path.join(config["OutputRoot"], "index.csv")
        if os.path.isfile(index_filepath):
            print(f"Using cached image crop index file at {index_filepath}, image cropping skipped")
            return

    if config["UsePreprocessing"]:
        cropper = ImageCropper(
            config["OutputRoot"],
            config["BaseInputDir"],
            dirs=config["Dirs"]
        )
        cropper.begin_cropping()

def train(config):
    # todo: implement UseCachedModel and model saving
    if (config["DoTraining"]):
        classifier = ResNetClassifier(config)
        classifier.load_data()
        classifier.train()

def full_train(config):
    preprocess(config["preprocessing"])
    train(config["training"])

def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    full_train(config)

if __name__ == "__main__":
    main()