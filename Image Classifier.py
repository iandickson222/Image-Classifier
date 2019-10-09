# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:20:16 2019

@author: Ian
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainSet = torchvision.datasets.FashionMNIST('./data', download = True, transform = transform, train = True)
testSet = torchvision.datasets.FashionMNIST('./data', download = True, transform = transform, train = False)

trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = 100, shuffle = True)
testLoader = torch.utils.data.DataLoader(testSet, batch_size = 100, shuffle = False)

#Neural Network
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = 5)
        self.fc1 = nn.Linear(in_features = 4*24*24, out_features = 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        
        x = x.view(-1, 4*24*24)
        x = self.fc1(x)
        x = F.log_softmax(x, dim = 1)
        
        return x

        

model = Model()
optimizer = optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.CrossEntropyLoss()

#Random Guessing
correct = 0
total = 0
with torch.no_grad():
    for data in testLoader:
        images, labels = data
        
        outputs = model(images)
        
        values, indices = torch.max(outputs, 1)
        
        correct += (indices == labels).sum().item()
        total += labels.size(0)
    print("Guessing Batch")
    print("Correct: {}".format(correct))
    print("Total: {}".format(total))
    print("Percent Correct: {}%".format(correct/total*100))
    print("\n")


# Train Network
for epoch in range(2):        
    correct = 0
    total = 0
    
    for data in trainLoader:
        images, labels = data
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        
        values , indices = torch.max(outputs, 1)

        correct += (indices == labels).sum().item()
        total += labels.size(0)
    
    print("Training Batch {}".format(epoch + 1))
    print("Correct: {}".format(correct))
    print("Total: {}".format(total))
    print("Percent Correct: {}%".format(correct/total*100))
    print("\n")


#Test Network
correct = 0
total = 0
with torch.no_grad():
    for data in testLoader:
        images, labels = data
        
        outputs = model(images)
        
        values, indices = torch.max(outputs, 1)
        
        correct += (indices == labels).sum().item()
        total += labels.size(0)
    print("Testing Batch")
    print("Correct: {}".format(correct))
    print("Total: {}".format(total))
    print("Percent Correct: {}%".format(correct/total*100))
    print("\n")


with torch.no_grad():
    catagories = {0: "Shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}
    
    data = next(iter(testLoader))
    random = np.random.randint(100)
    images, labels = data
    outputs = model(images)

    image = images[random][0]
    
    plt.imshow(image, cmap ="gray")
      
    value, index = torch.max(outputs[random], dim = 0)
    catagory = catagories[index.item()]
    print("Computer's Guess: {}".format(catagory))