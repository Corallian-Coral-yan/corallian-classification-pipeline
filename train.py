import os
import tomllib
from pathlib import Path
import logging
import wandb
from datetime import datetime

import numpy as np
import pandas as pd

from modules.model.classifier import ResNetASPPClassifier
from modules.preprocessing.image_cropper import ImageCropper
from modules.data.image_dataset import ImageDataset

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
    if train_config["DoTraining"] or test_config["DoTesting"]:
        if train_config["model"]["NumClasses"] == 'auto':
            train_config["model"]["NumClasses"] = get_num_classes(train_config["IndexFile"], train_config["model"]["LabelColumn"])
            logging.info(f"NumClasses is 'auto': detected {train_config["model"]["NumClasses"]} classes")

        classifier = ResNetASPPClassifier(train_config)
        classifier.load_data()
        logging.info("Classifier created")

        # Fix: Ensure labels are converted to LongTensor during training
        for batch_idx, (images, labels, _) in enumerate(classifier.train_loader):
            images = images.float().to(classifier.device)
            # Convert labels to LongTensor
            labels = labels.to(classifier.device).long()

            if batch_idx == 0:
                # Debug: Check label dtypes
                logging.info(f"Label dtype: {labels.dtype}")
                
        if train_config["LoadPretrainedModel"]:
            classifier.load_pretrained_model(train_config["PretrainedModelFilepath"], train_config["model"])
        else:
            logging.info("No pretrained model loaded")

        if train_config["DoTraining"]:
            classifier.train()
        else:
            logging.info("Training skipped")

        if test_config["DoTesting"]:
            test_output_folder = None if test_config["TestOutputFolder"] == "none" else test_config["TestOutputFolder"]
            classifier.test(load_best_model=train_config["DoTraining"], test_output_folder=test_output_folder)

        if train_config["SaveModel"] and train_config["DoTraining"]: 
            logging.info("Saving model. . .")
            classifier.save()

        

def get_num_classes(annotations_filepath, label_column, exclude=ImageDataset.AA_CLASSES_TO_IGNORE):
    df = pd.read_csv(annotations_filepath)
    classes = df[label_column].unique()
    classes[0] = "AA"

    logging.info(f"Unique classes found: {classes}")
    if exclude:
        classes = np.setdiff1d(classes, exclude)

        logging.info(f"Excluding classes {ImageDataset.AA_CLASSES_TO_IGNORE}")
        logging.info(f"Unique classes: {classes}")

    return len(classes)


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

    if logging_config["UseLogFile"]:
        artifact = wandb.Artifact('pipeline-logs', type='logs')
        artifact.add_file(logging_config["LogFile"])
        wandb.log_artifact(artifact)

if __name__ == "__main__":
    main()