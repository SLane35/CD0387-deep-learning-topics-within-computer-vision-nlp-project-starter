#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import argparse
import time
import os
import sys
import logging

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device):

    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss // len(test_loader)
    total_acc = running_corrects // len(test_loader)
    
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}\n".format(
            total_loss, total_acc)
    )

def train(model, train_loader, validation_loader, criterion, optimizer, epochs, device):
    
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(epochs):

        for phase in ['train', 'valid']:
            
            logger.info(f"Epoch {epoch}, Phase {phase}")
            
            if phase=='train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in image_dataset[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss // len(image_dataset[phase])
            epoch_acc = running_corrects // len(image_dataset[phase])
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1

            logger.info(
                "{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}\n".format(
                    phase, epoch_loss, epoch_acc, best_loss)
            )
            
        if loss_counter==1:
            break
            
    return model

    
def net():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 512),
                   nn.ReLU(inplace=True), nn.Linear(512,133))
    
    return model

def create_data_loaders(batch_size):
    
    train_dir = os.environ['SM_CHANNEL_TRAIN']
    val_dir = os.environ['SM_CHANNEL_VAL']
    test_dir=os.environ['SM_CHANNEL_TEST']
    

    training_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()])
    
    valid_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()])

    testing_transform = transforms.Compose([
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
  
    train_set = torchvision.datasets.ImageFolder(root=train_dir, transform=training_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.ImageFolder(root=test_dir, transform=testing_transform)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    validation_set = torchvision.datasets.ImageFolder(root=val_dir, transform=valid_transform)
    validation_loader  = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True) 
    
    return train_loader, test_loader, validation_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    model=net()
    model=model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    #train_dir = os.environ['SM_CHANNEL_TRAIN']
    #val_dir = os.environ['SM_CHANNEL_VAL']
    #test_dir=os.environ['SM_CHANNEL_TEST']
    print(args.batch_size)
    train_loader, test_loader, validation_loader = create_data_loaders(args.batch_size)
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, args.epochs, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    #test_loader = create_data_loaders(test_dir, batch_size)
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model.state_dict(), os.path.join(args.model_dir, "dogs-resnet18.pt"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.1, 
        metavar="LR", 
        help="learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 5)",
    )
    
    parser.add_argument("--model-dir", type=str, default=os.environ['SM_MODEL_DIR'])  
        
    args=parser.parse_args()
    
    
    main(args)
