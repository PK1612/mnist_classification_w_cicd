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
    
    # Enhanced hyperparameters
    BATCH_SIZE = 32  # Smaller batch size for better convergence
    LEARNING_RATE = 0.001
    
    # Load MNIST dataset with augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomRotation(10),  # Add slight rotation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))  # Add slight shift
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    # Train with multiple passes over the data
    best_accuracy = 0
    best_model_path = None
    
    print("Starting training...")
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Validate every 100 batches
        if batch_idx % 100 == 0:
            model.eval()
            correct = 0
            total = 0
            val_loss = 0
            
            # Quick validation on a subset of test data
            test_dataset = datasets.MNIST('./data', train=False, 
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
            
            with torch.no_grad():
                for val_data, val_target in test_loader:
                    val_data, val_target = val_data.to(device), val_target.to(device)
                    val_output = model(val_data)
                    val_loss += criterion(val_output, val_target).item()
                    _, predicted = torch.max(val_output.data, 1)
                    total += val_target.size(0)
                    correct += (predicted == val_target).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join('models', f'mnist_model_{timestamp}_{accuracy:.2f}.pth')
                os.makedirs('models', exist_ok=True)
                torch.save(model.state_dict(), save_path)
                best_model_path = save_path
                
                if accuracy > 95:
                    print(f"Achieved target accuracy: {accuracy:.2f}%")
                    return best_model_path
            
            scheduler.step(val_loss)
            model.train()
    
    if best_model_path is None:
        raise Exception("Failed to achieve target accuracy of 95%")
    
    return best_model_path

if __name__ == "__main__":
    train() 