import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Attempting to run ResNet18 with HDC on device {device}")

torch.zeros(1).cuda()

ANNOTATIONS_FILE = r"./dataset/CROPPED-CORALS/index.csv"
IMG_DIR = r"./dataset/CROPPED-CORALS/"

num_classes = 34   # 34 classes detected for initial dataset
num_epochs = 20
batch_size = 16
learning_rate = 0.01

model_verbose = False

from model.resnet import ResNet, ResidualBlock
from model.resnet_hdc import ResNet18_HDC, ResNet101_HDC

model = ResNet18_HDC(num_classes=num_classes, verbose=model_verbose).to(device)
print(f"Model architecture: {str(model)}")

from dataset.image_dataset import ImageDataset

def data_loader(batch_size, random_seed=42, valid_size=0.1, shuffle=True, test=False):
    # define transforms
    transform = transforms.ToTensor()
    target_transform = None

    if test:
        dataset = ImageDataset(
            ANNOTATIONS_FILE, 
            IMG_DIR, 
            transform=transform, 
            target_transform=target_transform, 
            random_state=random_seed
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader

    # load the dataset
    train_dataset = ImageDataset(ANNOTATIONS_FILE, IMG_DIR, train=True, transform=transform, target_transform=target_transform, random_state=random_seed)
    valid_dataset = ImageDataset(ANNOTATIONS_FILE, IMG_DIR, train=True, transform=transform, target_transform=target_transform, random_state=random_seed)

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

train_loader, valid_loader = data_loader(batch_size=batch_size)
test_loader = data_loader(batch_size=batch_size, test=True)
print("Successfully created data loaders")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)  

# Train the model
import gc
total_step = len(train_loader)

print("Beginning training")
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        print(f"Epoch {epoch + 1}/{num_epochs} | Batch {i}")
        
        #Move tensors to the configured device
        images = images.float()
        images = images.to(device)
        labels = labels.long()
        labels = labels.to(device)

        #Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del images, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()

    print ('Epoch [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, num_epochs, loss.item()))

#Validation
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs

    print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))
