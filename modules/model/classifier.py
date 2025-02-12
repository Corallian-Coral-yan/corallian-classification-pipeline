import gc

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from modules.model.resnet_hdc import ResNet18_HDC, ResNet101_HDC
from modules.model.aspp import ASPP
from modules.data.image_dataset import ImageDataset

class ResNetASPPClassifier(nn.Module):
    def __init__(self, config):
        super(ResNetASPPClassifier, self).__init__()
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.config["ForceUseCuda"]:
            self.assert_has_cuda()

        # print(f"Attempting to run ResNet{self.config["ResNetModel"]} with HDC on device {self.device}")

        self.annotations_file = self.config["IndexFile"]
        self.img_dir = self.config["ImageDir"]
            
        model_config = self.config["model"]
        self.num_classes = model_config["NumClasses"]
        self.num_epochs = model_config["NumEpochs"]
        self.batch_size = model_config["BatchSize"]
        self.model_verbose = model_config["Verbose"]
        self.random_seed = model_config["RandomSeed"]
        self.validation_split = model_config["ValidationSplit"]

        if model_config["LossFunction"] == "cross-entropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise TypeError(f"Invalid Loss Function: {model_config["LossFunction"]}")

        # Choose ResNet model
        if self.config["ResNetModel"] == 18:
            resnet_model = ResNet18_HDC
        elif self.config["ResNetModel"] == 101:
            resnet_model = ResNet101_HDC
        else:
            raise TypeError("Invalid ResNet Configuration, must be 18 or 101")
        
        if self.config["UseCachedModel"]:
            self.model = torch.load(self.config["ModelFilepath"], weights_only=False)
            print(f"Using cached model at {self.config["ModelFilepath"]}")
        else:
            self.model = resnet_model(num_classes=self.num_classes, verbose=self.model_verbose).to(self.device)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-2])  # Keep only backbone
        
        # ASPP settings
        aspp_config = model_config["aspp"]
        self.aspp_enabled = aspp_config.get("ASPPEnabled", True) 
        self.aspp_in_channels = aspp_config.get("ASPPInChannels", 1024)
        self.aspp_out_channels = aspp_config.get("ASPPOutChannels", 256)
        self.atrous_rates = aspp_config.get("AtrousRates", [6, 12, 18])
        
        # ASPP model
        if self.aspp_enabled:
            print(
                f"Using ASPP with {self.aspp_in_channels} in channels and {self.aspp_out_channels} out channels and rates {self.atrous_rates}")
            self.aspp = ASPP(in_channels=self.aspp_in_channels, out_channels=self.aspp_out_channels, atrous_rates=self.atrous_rates)
        
        # Global Average Pooling + Fully Connected Classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
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
            raise TypeError(f"Invalid Optimizer: {optim_config["OptimizerName"]}")
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.aspp(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def load_data(self):
        self.train_loader, self.valid_loader = self._data_loader(batch_size=self.batch_size, random_seed=self.random_seed, valid_size=self.validation_split)
        self.test_loader = self._data_loader(batch_size=self.batch_size, random_seed=self.random_seed, test=True)
        print("Successfully created data loaders")

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
        print("Beginning training")
        
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):  
                print(f"Epoch {epoch + 1}/{self.num_epochs} | Batch {i + 1}/{total_step}")
                
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

            print ('Epoch [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, self.num_epochs, loss.item()))

        #Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.valid_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

            print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))

        if self.config["SaveModel"]:
            self.save()

    def save(self):
        torch.save(self.model, self.config["ModelFilepath"])