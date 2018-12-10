import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import os
import torch.optim as optim
import argparse

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


def test(model, dataloader, num_classes, classes, batch=256, device="cuda"):
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


def train(model, trainloader, testloader, batch_size, num_epoch, criterion, optimizer, log_step, num_classes, classes,
          model_save_dir, log_save_dir, device="cuda", save_step=160):
    model.to(device)
    
    logs = {"trn_metrics": {"train_loss": [], "train_accu": []}, "tst_metrics": {"test_accu": []},
            "meta": {"log_step": log_step, "train_bsize": batch_size}}
    
    for epoch in range(num_epoch):
        
        print("****************** Begin training epoch: {} ********************".format(epoch + 1))
        
        train_logs = train_one_epo(model, trainloader, criterion, optimizer, log_step, device=device)
        
        test_logs = test(model, testloader, num_classes, classes=classes, device=device)
        for k, v in train_logs.items():
            logs["trn_metrics"][k].append(v)
            
        for k, v in test_logs.items():
            logs["tst_metrics"][k].append(v)
            
        if (epoch + 1) % save_step == 0:
            save_model(model, os.path.join(model_save_dir, "ANet_no_pre_{}.pt".format(epoch)))
            save_log(logs, os.path.join(log_save_dir, 'log_ANet_no_pre.json'))
    
    print('Finished Training')
    return logs


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes=4, feature_extract=True, use_pretrained=True):
    model_ft = None
    if model_name == "inception":
        """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--traditional_aug", type=int, default=0)
    parser.add_argument("--neural_aug", type=int, default=0)
    parser.add_argument("--log_save_dir")
    parser.add_argument("--model_save_dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--save_step", type=int, default=160)
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = "/data/dataset-of-4"
    data_cat = ('train', 'train-after-10000', 'val')
    
    input_shape = (299, 299)
    batch_size = 64
    use_parallel = True
    use_gpu = True
    num_output = 4
    
    pre_data_transforms = transforms.Compose([
        transforms.Resize(input_shape),
        transforms.ToTensor()])
    
    pre_image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), pre_data_transforms) for x in data_cat}
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
    
    if args.traditional_aug:
        data_transforms = transforms.Compose([
                transforms.RandomResizedCrop(input_shape[0]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
  
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in data_cat}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=4) for x in data_cat}
    dataset_sizes = {x: len(image_datasets[x]) for x in data_cat}
    train_before, train_after, val = data_cat

    if args.neural_aug:
        trainloader, testloader = dataloaders[train_after], dataloaders[val]
    else:
        trainloader, testloader = dataloaders[train_before], dataloaders[val]
        
    classes = image_datasets[train_before].classes
    
    print("The number of data before augumentation: {}".format(len(image_datasets[train_before])))
    print("The number of data after neural augumentation: {}".format(len(image_datasets[train_after])))
    
#    net = torchvision.models.inception_v3(pretrained=False, aux_logits=False, num_classes=num_output)
    net = torchvision.models.inception_v3()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    logs = train(model=net, trainloader=trainloader, testloader=testloader, batch_size=batch_size,
                 num_epoch=args.epochs, criterion=criterion, optimizer=optimizer, num_classes=4, log_step=20,
                 classes=classes, device=device, model_save_dir=args.model_save_dir, log_save_dir=args.log_save_dir, save_step=args.save_step)

