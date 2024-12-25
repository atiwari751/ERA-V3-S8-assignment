from __future__ import print_function
import torch
import torchvision
from torchvision import datasets, transforms
import numpy
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm
from model import Net, get_device
from tabulate import tabulate
import torch.nn.functional as F
import torch.nn as nn

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Training Parameters
EPOCHS = 20
batch_size_train = 128
batch_size_test = 1000

# CIFAR10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Get initial parameters
        if batch_idx == 0:
            initial_params = next(model.parameters())[0,0,0,0].item()
        
        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        

        # Update Progress Bar
        pred = pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc=f'Epoch {epoch} | Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    return 100*correct/processed, train_loss/len(train_loader)

def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Store misclassified samples
            misclassified_mask = ~pred.eq(target.view_as(pred)).squeeze()
            if misclassified_mask.any():
                misclassified_images.append(data[misclassified_mask])
                misclassified_labels.append(target[misclassified_mask])
                misclassified_preds.append(pred[misclassified_mask])

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy:.2f}%)\n')
    
    return test_accuracy, test_loss, misclassified_images, misclassified_labels, misclassified_preds

def plot_misclassified(images, labels, preds, classes):
    fig = plt.figure(figsize=(20, 10))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.tight_layout()
        plt.imshow(images[i].cpu().squeeze().permute(1, 2, 0))
        plt.title(f'Predicted: {classes[preds[i]]}\nActual: {classes[labels[i]]}')
        plt.xticks([])
        plt.yticks([])
    plt.show()

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using Device: {device}")
    
    # Load the data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                             shuffle=True, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                            shuffle=False, num_workers=4, pin_memory=True)

    # Initialize the model, optimizer and criterion
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=0.001, 
                                weight_decay=1e-4)
    criterion = nn.NLLLoss()

    # Lists to store metrics for plotting
    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        train_acc, train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        test_acc, test_loss, misclassified_images, misclassified_labels, misclassified_preds = test(model, device, test_loader, criterion, epoch)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    # Print final results in Excel-friendly format
    print("\nTraining Summary:")
    print("Epoch\tTraining Accuracy\tTest Accuracy")  # Tab-separated headers
    for epoch in range(1, EPOCHS + 1):
        print(f"{epoch}\t{train_accuracies[epoch-1]:.2f}\t{test_accuracies[epoch-1]:.2f}")

    # Plot misclassified images from the last epoch
    print("\nDisplaying 10 misclassified images from the last epoch:")
    misclassified_images = torch.cat(misclassified_images)
    misclassified_labels = torch.cat(misclassified_labels)
    misclassified_preds = torch.cat(misclassified_preds)
    plot_misclassified(misclassified_images[:10], 
                      misclassified_labels[:10], 
                      misclassified_preds[:10], 
                      classes)

    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
