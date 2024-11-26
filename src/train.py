import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
from datetime import datetime
import os

def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train for one epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
        # Quick validation every 200 batches
        if batch_idx % 200 == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for val_data, val_target in torch.utils.data.DataLoader(
                    datasets.MNIST('./data', train=False, transform=transform),
                    batch_size=1000):
                    val_data, val_target = val_data.to(device), val_target.to(device)
                    val_output = model(val_data)
                    _, predicted = torch.max(val_output.data, 1)
                    total += val_target.size(0)
                    correct += (predicted == val_target).sum().item()
                print(f'Current Accuracy: {100 * correct / total:.2f}%')
            model.train()
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join('models', f'mnist_model_{timestamp}.pth')
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return save_path

if __name__ == "__main__":
    train() 