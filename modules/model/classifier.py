import gc
import logging

import numpy as np
import torch
import torch.nn as nn
from .metrics import compute_metrics
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

        # ResNet backbone
        if self.config["ResNetModel"] == 18:
            resnet_model = ResNet18_HDC
        elif self.config["ResNetModel"] == 101:
            resnet_model = ResNet101_HDC
        else:
            raise TypeError("Invalid ResNet Configuration, must be 18 or 101")
        
        # Cached model
        if self.config["UseCachedModel"]:
            logging.info(f"Loading cached model from {self.config['ModelFilepath']}...")
            self.model = torch.load(self.config["ModelFilepath"], weights_only=False)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            logging.info("Model loaded successfully!")
            return
        else:
            self.model = resnet_model(num_classes=self.num_classes, verbose=self.model_verbose).to(self.device)
            
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
            self.aspp_out_channels = 512  # Default output channels from ResNet
        # Visual Embedding
        self.visual_embedding_enabled = self.config["model"]["visual_embedding"].get("EmbeddingEnabled", True)
    
        if self.visual_embedding_enabled:
            logging.info(f"Using Visual Embedding with {self.aspp_out_channels} in channels and 256 out channels")
            self.visual_embedding = VisualEmbedding(in_channels=self.aspp_out_channels, embedding_dim=256).to(self.device)

        # Global Average Pooling + Fully Connected Classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.aspp_out_channels, self.num_classes)

        optim_config = model_config["optimizer"]
        if optim_config["OptimizerName"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=optim_config["SGDLearningRate"], 
                weight_decay = optim_config["SGDWeightDecay"], 
                momentum = optim_config["SGDMomentum"]
            )  
        else:
            raise TypeError(f"Invalid Optimizer: {optim_config['OptimizerName']}")
        
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
        # Train the model
        total_step = len(self.train_loader)
        logging.info("Beginning training")
        
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):  
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
                
                del images, labels, outputs
                torch.cuda.empty_cache()
                gc.collect()

            logging.info ('Epoch [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, self.num_epochs, loss.item()))

        #Validation
        self.evaluate(self.valid_loader, "valid_loader")
              
        if self.config["SaveModel"]:
            logging.info("Saving model. . .")
            self.save()

    def save(self):
        torch.save(self.model, self.config["ModelFilepath"])

    def evaluate(self, data_loader, name=""):
        logging.info(f"Running evaluation on data loader {name}")

        with torch.no_grad():
            y_true = []
            y_pred = []
            correct = 0
            total = 0

            total_step = len(data_loader)
            for i, (images, labels) in enumerate(data_loader):
                logging.info(f"Evaluating | Batch {i + 1}/{total_step}")
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

                del images, labels, outputs

            logging.info("y_true:", y_true[:10])  # Show first 10 labels
            logging.info("y_pred:", y_pred[:10])  # Show first 10 predictions
            logging.info("Unique labels in y_true:", set(y_true))
            logging.info("Unique labels in y_pred:", set(y_pred))

            accuracy = 100 * correct / total
            logging.info(f'Accuracy: {accuracy:.2f}%')

            # Compute precision, recall, F1-score
            metrics = compute_metrics(y_true, y_pred)
            logging.info("Metrics dictionary:", metrics)  # Debugging step
            logging.info(f'Precision: {metrics["precision"]:.4f}')
            logging.info(f'Recall: {metrics["recall"]:.4f}')
            logging.info(f'F1 Score: {metrics["f1_score"]:.4f}')

    def validate(self):
        self.evaluate(self.valid_loader, "valid_loader")  

    def test(self):
        return self.evaluate(self.test_loader, "test_loader")