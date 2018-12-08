import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from os import path

# import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = "/data/dataset-of-4"

data_cat = ('train', 'train-after-10000', 'val')

input_shape = (299, 299)
batch_size = 64
# scale = 360
use_parallel = True
use_gpu = True
epochs = 100
num_output = 4

pre_data_transforms = transforms.Compose([
        transforms.Resize(input_shape),
        transforms.ToTensor()])

pre_image_datasets = {x: datasets.ImageFolder(path.join(data_dir, x), pre_data_transforms) for x in data_cat}
pre_dataloaders = {x: torch.utils.data.DataLoader(pre_image_datasets[x], batch_size=batch_size,
                                         shuffle=True, num_workers=4) for x in data_cat}

train_before, train_after, val = data_cat

pre_trainloader, pre_testloader = pre_dataloaders[train_after], pre_dataloaders[val]

mean = 0.
std = 0.
nb_samples = 0.
for data, labels in pre_trainloader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

# no traditional augmentation
data_transforms = transforms.Compose([
        transforms.Resize(input_shape),
        transforms.ToTensor(),
transforms.Normalize(mean, std)])

# traditional augmentation
# data_transforms = transforms.Compose([
#         transforms.RandomResizedCrop(input_shape[0]),
#     transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
# transforms.Normalize(mean, std)])

image_datasets = {x: datasets.ImageFolder(path.join(data_dir, x), data_transforms) for x in data_cat}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                              shuffle=True, num_workers=4) for x in data_cat}
dataset_sizes = {x: len(image_datasets[x]) for x in data_cat}
train_before, train_after, val = data_cat

# with neural
trainloader, testloader = dataloaders[train_after], dataloaders[val]

# no neural
# trainloader, testloader = dataloaders[train_before], dataloaders[val]

classes = image_datasets[train_before].classes

print("The number of data before augumentation: {}".format(len(image_datasets[train_before])))
print("The number of data after neural augumentation: {}".format(len(image_datasets[train_after])))

net = torchvision.models.inception_v3(pretrained=False, aux_logits = False, num_classes=num_output)


def train_one_epo(model, dataloader, criterion, optimizer, log_step, device="cuda"):
    logs = {"train_loss": [], "train_accu": []}
    
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # print statistics
        running_loss += loss.item()
        if i % log_step == log_step - 1:
            print('loss for [%d, %d] batch: %.3f' %
                  (i + 1 - log_step, i + 1, running_loss / log_step))
            logs["train_loss"].append(running_loss / log_step)
            running_loss = 0.0
    
    #         print(correct, total)
    print('Accuracy of the network on the {} training images: %d %%'.format(total) % (
            100 * correct / total))
    logs["train_accu"] = 100 * correct / total
    
    return logs


def test(model, dataloader, num_classes, batch=256, device="cuda"):
    model.to(device)
    
    logs = {"test_accu": []}
    
    correct = 0
    total = 0
    
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            c = (predicted == labels).squeeze()
            
            for i in range(batch):
                if i >= len(labels):
                    break
                
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    #     print(correct, total)
    print('Accuracy of the network on the {} test images: %d %%'.format(total) % (
            100 * correct / total))
    logs["test_accu"].append(100 * correct / total)
    
    for i in range(num_classes):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    
    return logs


def save_model(model, path):
    torch.save(model.state_dict(), path)


def save_log(log, path):
    import json
    
    jsObj = json.dumps(log)
    
    fileObject = open(path, 'w+')
    fileObject.write(jsObj)
    fileObject.close()


def train(model, trainloader, testloader, batch_size, num_epoch, criterion, optimizer, log_step, num_classes,
          device="cuda"):
    model.to(device)
    
    logs = {"trn_metrics": {"train_loss": [], "train_accu": []}, "tst_metrics": {"test_accu": []},
            "meta": {"log_step": log_step, "train_bsize": batch_size}}
    
    for epoch in range(num_epoch):
        
        print("****************** Begin training epoch: {} ********************".format(epoch + 1))
        
        train_logs = train_one_epo(model, trainloader, criterion, optimizer, log_step, device=device)
        test_logs = test(model, testloader, num_classes, device=device)
        
        save_model(model, "ANet_no_pre_{}.pt".format(epoch))
        
        for k, v in train_logs.items():
            logs["trn_metrics"][k].append(v)
        
        for k, v in test_logs.items():
            logs["tst_metrics"][k].append(v)
        
        save_log(logs, 'log_ANet_no_pre.json')
    
    print('Finished Training')
    return logs

epoch = 2

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())


logs = train(net, trainloader, testloader, batch_size, epoch, criterion, optimizer, num_classes = 4, log_step=20, device=device)


# # print(logs)
# log_path = 'log_ANet_no_pre.json'

# jsObj = json.dumps(logs)

# fileObject = open(log_path, 'w+')
# fileObject.write(jsObj)
# fileObject.close()