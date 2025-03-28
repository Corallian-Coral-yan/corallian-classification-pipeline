import gc
import logging
import itertools
import math
import os
import traceback

import wandb
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import wandb.sklearn
from .metrics import compute_metrics, compute_confusion_matrix
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from modules.model.resnet_hdc import ResNet18_HDC, ResNet101_HDC
from modules.model.aspp import ASPP
from modules.data.image_dataset import ImageDataset
from modules.model.visual_embeddings import VisualEmbedding

class ResNetASPPClassifier(nn.Module):
    def __init__(self, config):
        super(ResNetASPPClassifier, self).__init__()
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Running classifier on device {self.device}")

        if self.config["ForceUseCuda"]:
            self.assert_has_cuda()

        self.annotations_file = self.config["IndexFile"]
        self.img_dir = self.config["ImageDir"]
            
        model_config = self.config["model"]
        self.num_classes = model_config["NumClasses"]
        self.num_epochs = model_config["NumEpochs"]
        self.batch_size = model_config["BatchSize"]
        self.model_verbose = model_config["Verbose"]
        self.random_seed = model_config["RandomSeed"]
        self.validation_split = model_config["ValidationSplit"]

        # Loss function
        if model_config["LossFunction"] == "cross-entropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise TypeError(f"Invalid Loss Function: {model_config['LossFunction']}")

        # Image Transforms
        self.grayscale = model_config["ReadInputAsGrayscale"]

        # ResNet backbone
        if self.config["ResNetModel"] == 18:
            resnet_model = ResNet18_HDC
        elif self.config["ResNetModel"] == 101:
            resnet_model = ResNet101_HDC
        else:
            raise TypeError("Invalid ResNet Configuration, must be 18 or 101")
        
        self.model = resnet_model(
            num_classes=self.num_classes,
            verbose=self.model_verbose,
            in_channels=1 if self.grayscale else 3
        ).to(self.device)
            
        # Extract feature maps from ResNet (excluding last classification layers)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-2])  # Keep only backbone
        
        # ASPP settings
        aspp_config = model_config["aspp"]
        self.aspp_enabled = aspp_config.get("ASPPEnabled", True) 
        
        # ASPP model
        if self.aspp_enabled:
            logging.info("ASPP Enabled. Loading ASPP model. . .")
            self.aspp_in_channels = aspp_config.get("ASPPInChannels", 1024)
            self.aspp_out_channels = aspp_config.get("ASPPOutChannels", 256)
            self.atrous_rates = aspp_config.get("AtrousRates", [6, 12, 18])

            logging.info(f"Using ASPP with {self.aspp_in_channels} in channels and {self.aspp_out_channels} out channels and rates {self.atrous_rates}")
            self.aspp = ASPP(in_channels=self.aspp_in_channels, out_channels=self.aspp_out_channels, atrous_rates=self.atrous_rates).to(self.device)
        else:
            logging.info("ASPP Disabled. . .")
            self.aspp_out_channels = 2048  # Default output channels from ResNet
        # Visual Embedding
        self.visual_embedding_enabled = self.config["model"]["visual_embedding"].get("EmbeddingEnabled", True)
    
        if self.visual_embedding_enabled:
            logging.info(f"Using Visual Embedding with {self.aspp_out_channels} in channels and 256 out channels")
            self.visual_embedding = VisualEmbedding(in_channels=self.aspp_out_channels, embedding_dim=256).to(self.device)

        # Global Average Pooling + Fully Connected Classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1)).to(self.device)
        self.fc = nn.Linear(self.aspp_out_channels, self.num_classes).to(self.device)

        optim_config = model_config["optimizer"]
        if optim_config["OptimizerName"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=optim_config["SGDLearningRate"], 
                weight_decay = optim_config["SGDWeightDecay"], 
                momentum = optim_config["SGDMomentum"]
            ) 
        elif optim_config["OptimizerName"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=optim_config["AdamLearningRate"],
                weight_decay=optim_config["AdamWeightDecay"]
            )
        elif optim_config["OptimizerName"] == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=optim_config["AdamWLearningRate"],
                weight_decay=optim_config["AdamWWeightDecay"]
            )
        else:
            raise TypeError(f"Invalid Optimizer: {optim_config['OptimizerName']}")
        
        # Checkpointing Settings
        self.use_checkpoints = False
        self.load_checkpoints = False

        self.start_epoch = 1
        self.start_checkpoint = 0
        self.checkpoints_per_epoch = 1
        self.checkpoint_folder = None

        checkpoint_config = config["checkpoint"]
        if checkpoint_config["UseCheckpoints"]:
            self.use_checkpoints = True

            self.checkpoints_per_epoch = checkpoint_config["CheckpointsPerEpoch"]
            self.checkpoint_folder = checkpoint_config["CheckpointFolder"]

            if not os.path.isdir(self.checkpoint_folder):
                os.mkdir(self.checkpoint_folder)

            if checkpoint_config["LoadCheckpoint"]:
                self.load_checkpoints = True

                self.start_epoch = checkpoint_config["StartEpoch"]
                self.start_checkpoint = checkpoint_config["StartCheckpoint"]

                # load state_dict back into memory
                self.resume_from_checkpoint(self.start_epoch, self.start_checkpoint)    

        # variables for crash recovery
        self.current_epoch = 0
        self.current_checkpoint = 0            
        
    def forward(self, x):
        x = self.feature_extractor(x)
        
        if self.aspp_enabled:
            x = self.aspp(x)

        if self.visual_embedding_enabled:
            x = self.visual_embedding(x)
            x = x.view(x.shape[0], x.shape[1], 1, 1)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def load_data(self):
        self.train_loader, self.valid_loader = self._data_loader(batch_size=self.batch_size, random_seed=self.random_seed, valid_size=self.validation_split)
        self.test_loader = self._data_loader(batch_size=self.batch_size, random_seed=self.random_seed, test=True)
        logging.info("Successfully created data loaders")

    def _data_loader(self, batch_size, random_seed=42, valid_size=0.1, shuffle=True, test=False):
        # define transforms
        if self.grayscale:
            transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        else:
            transform = transforms.ToTensor()
        target_transform = None

        if test:
            dataset = ImageDataset(
                self.annotations_file, 
                self.img_dir, 
                transform=transform, 
                target_transform=target_transform, 
                random_state=random_seed
            )

            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle
            )

            return data_loader

        # load the dataset
        train_dataset = ImageDataset(self.annotations_file, self.img_dir, train=True, transform=transform, target_transform=target_transform, random_state=random_seed)
        valid_dataset = ImageDataset(self.annotations_file, self.img_dir, train=True, transform=transform, target_transform=target_transform, random_state=random_seed)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler)

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler)

        return (train_loader, valid_loader)

    def assert_has_cuda(self):
        torch.zeros(1).cuda()

    def train(self):
        if self.use_checkpoints:
            max_retries = self.config["checkpoint"]["MaxRetriesOnECCError"]
            if max_retries == "none":
                max_retries = None
            
        retries = 0

        while max_retries is None or max_retries < retries:
            try:
                # load the latest checkpoint, but only if already retried
                if retries > 0 and self.current_checkpoint != 0 and self.current_epoch != 0:
                    logging.info(f"{self.current_checkpoint} {self.current_epoch}")

                    self.resume_from_checkpoint(self.current_epoch, self.current_checkpoint)
                    self.start_epoch = self.current_epoch
                    self.start_checkpoint = self.current_checkpoint

                self._train()
            except RuntimeError as e:
                if "uncorrectable ecc error encountered" in str(e).lower():
                    if self.use_checkpoints and max_retries is not None and retries == max_retries:
                        logging.fatal(f"ECC errors encountered {retries + 1} time(s), quitting")
                        raise e
                    else:
                        logging.error(f"ECC errors encountered {retries + 1} time(s), restarting")
                        logging.error(traceback.format_exc())
                else:
                    raise e
            finally:
                retries += 1
        
    def _train(self):            
        # Train the model
        total_step = len(self.train_loader)
        logging.info("Beginning training")
        
        checkpoint_step = math.ceil(total_step / self.checkpoints_per_epoch)
        
        # workaround for ensuring checkpoints don't crash
        loss = None

        for epoch in range(self.start_epoch - 1, self.num_epochs):
            checkpoint_batch_iter = itertools.count(start=checkpoint_step, step=checkpoint_step)
            next_checkpoint_batch = checkpoint_batch_iter.__next__()

            # set next checkpoint batch based on starting checkpoint
            if self.use_checkpoints and epoch == self.start_epoch - 1:
                for _ in range(self.start_checkpoint):
                    next_checkpoint_batch = checkpoint_batch_iter.__next__()

            for i, (images, labels) in enumerate(self.train_loader):                  
                if (self.use_checkpoints 
                        and epoch == self.start_epoch - 1 
                        and i < (self.start_checkpoint * checkpoint_step)):
                    
                    continue

                logging.info(f"Epoch {epoch + 1}/{self.num_epochs} | Batch {i + 1}/{total_step}")
                
                # Move tensors to the configured device
                images = images.float().to(self.device)
                labels = labels.to(self.device).long()

                # Forward pass
                outputs = self(images).float()
                
                # outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                wandb.log({
                    "epoch": epoch,
                    "batch": i,
                    "loss": loss.item()
                })

                del images, labels, outputs
                torch.cuda.empty_cache()
                gc.collect()

                # Generate checkpoint if enough epochs have passed
                if self.config["checkpoint"]["UseCheckpoints"] and (i + 1 >= next_checkpoint_batch or i == total_step - 1):
                    checkpoint_number = next_checkpoint_batch // checkpoint_step
                    checkpoint_filename = self.create_checkpoint(epoch + 1, checkpoint_number)
                    next_checkpoint_batch = checkpoint_batch_iter.__next__()

                    self.current_epoch = epoch + 1
                    self.current_checkpoint = checkpoint_number

                    logging.info(f"Generated checkpoint at {checkpoint_filename}")


            if loss:
                logging.info ('Epoch [{}/{}], Loss: {:.4f}' 
                            .format(epoch+1, self.num_epochs, loss.item()))
            

        #Validation
        self.evaluate(self.valid_loader, "valid_loader")
              
        if self.config["SaveModel"]:
            logging.info("Saving model. . .")
            self.save()

    def save(self):
        torch.save(self, self.config["PretrainedModelFilepath"])

    def create_checkpoint(self, epoch_number, checkpoint_number):
        filepath = os.path.join(
            self.checkpoint_folder, 
            f"Epoch{epoch_number}-Checkpoint{checkpoint_number}.pt"
        )
        torch.save(self.state_dict(), filepath)
        logging.info(f"Successfully saved state_dict to checkpoint {filepath}")
        return filepath
    
    def resume_from_checkpoint(self, epoch_number, checkpoint_number):
        filepath = os.path.join(
            self.checkpoint_folder, 
            f"Epoch{epoch_number}-Checkpoint{checkpoint_number}.pt"
        )
        self.load_state_dict(torch.load(filepath))
        logging.info(f"Successfully loaded state_dict from checkpoint {filepath}")

    def load_pretrained_model(self, filepath):
        model = torch.load(filepath, weights_only=False)
        self.load_state_dict(model.state_dict())
        logging.info(f"Successfully loaded state_dict from pretrained model {filepath}")

    def evaluate(self, data_loader, name=""):
        logging.info(f"Running evaluation on data loader {name}")

        with torch.no_grad():
            correct = 0
            total = 0

            y_true = []  # Store true labels
            y_pred = []  # Store predicted labels

            total_step = len(data_loader)
            for i, (images, labels) in enumerate(data_loader):
                logging.info(f"Evaluating | Batch {i + 1}/{total_step}")
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                y_true.extend(labels.cpu().numpy())  
                y_pred.extend(predicted.cpu().numpy())

                del images, labels, outputs

            accuracy = 100 * correct / total

            wandb.log({
                f"{name}_accuracy": accuracy
            })

            logging.info(f'{name}_Accuracy: {accuracy:.2f}%')

            if name == "valid_loader":
                dataset = self.valid_loader.dataset
                idx_to_class = dataset.idx_to_class
            else:
                dataset = self.test_loader.dataset
                idx_to_class = dataset.idx_to_class

            # Compute confusion matrix and pass it to formatter
            conf_matrix =  compute_confusion_matrix(y_true, y_pred)
            
            labels = [idx_to_class[i] for i in range(conf_matrix.shape[0])]
            self.format_confusion_matrix(conf_matrix, labels, name=name)

    def validate(self):
        self.evaluate(self.valid_loader, "valid_loader")  

    def test(self):
        return self.evaluate(self.test_loader, "test_loader")
    
    def format_confusion_matrix(self, conf_matrix, labels, name=""):
        df = pd.DataFrame(conf_matrix, index=[f"True_{label}" for label in labels], columns=[f"Pred_{label}" for label in labels])
        logging.info(f"\n{name} Confusion Matrix:\n{df.to_string()}")

        # Log confusion matrix to W&B as table and artifact        
        table = wandb.Table(dataframe=df)
        wandb.log({f"{name}_confusion_matrix_table": table})
        df.to_csv(f"{name}_confusion_matrix.csv")

        artifact = wandb.Artifact(f"{name}-confusion-matrix", type="confusion_matrix")
        artifact.add_file(f"{name}_confusion_matrix.csv")
        wandb.log_artifact(artifact)
