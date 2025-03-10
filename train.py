import os
import tomllib
from pathlib import Path
import logging
import wandb
from datetime import datetime

import pandas as pd

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
            logging.info(f"Using cached image crop index file at {index_filepath}, image cropping skipped")
            return

    if config["UsePreprocessing"]:
        cropper = ImageCropper(
            config["OutputRoot"],
            config["BaseInputDir"],
            dirs=config["Dirs"],
            recurse=config["InputRecurse"]
        )
        cropper.begin_cropping()

def train(train_config, test_config):
    if train_config["DoTraining"] or train_config["DoValidation"] or test_config["DoTesting"]:
        if train_config["model"]["NumClasses"] == 'auto':
            train_config["model"]["NumClasses"] = get_num_classes(train_config["IndexFile"])
            logging.info(f"NumClasses is 'auto': detected {train_config["model"]["NumClasses"]} classes")

        classifier = ResNetASPPClassifier(train_config)
        classifier.load_data()
        logging.info("Classifier created")

        # Fix: Ensure labels are converted to LongTensor during training
        for batch_idx, (images, labels) in enumerate(classifier.train_loader):
            images = images.float().to(classifier.device)
            # Convert labels to LongTensor
            labels = labels.to(classifier.device).long()

            if batch_idx == 0:
                # Debug: Check label dtypes
                logging.info(f"Label dtype: {labels.dtype}")

        # Loading cached model is for eval purposes, not for retraining
        if not train_config["UseCachedModel"]:
            classifier.train()

        if train_config["DoValidation"]:
            classifier.validate()     # Run evaluation after loading cached model

        if test_config["DoTesting"]:
            classifier.test()
        

def get_num_classes(annotations_filepath):
    df = pd.read_csv(annotations_filepath)
    return len(df['annotation'].unique())


def full_train(config):
    preprocess(config["preprocessing"])
    train(config["training"], config["testing"])

    logging.info("Done")

def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    timestamp = datetime.now().strftime("%m%d%y-%H")
    wandb.login(key=config['wandb']['WANDB_API_KEY'], relogin=config['wandb']['relogin'])
    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'], 
        name = f"{config['wandb']['runname']}-{timestamp}", # Set run name
        config=config, # Set config file
    )
    logging_config = config["logging"]

    logging.basicConfig(
        format=logging_config["LogFormat"], 
        level=logging.DEBUG, 
        filename=logging_config["LogFile"] if logging_config["UseLogFile"] else None,
        filemode='w'
    )

    full_train(config)

if __name__ == "__main__":
    main()