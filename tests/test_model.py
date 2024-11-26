import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MNISTModel
import torch
import pytest
import torch.nn.utils.prune as prune

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_parameters():
    model = MNISTModel()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {num_params}")  # Debug print
    assert num_params < 25000, f"Model has too many parameters: {num_params}"

def test_input_output_shape():
    model = MNISTModel()
    # Create a single test image: [batch_size=1, channels=1, height=28, width=28]
    test_input = torch.randn(1, 1, 28, 28)
    
    # Forward pass
    output = model(test_input)
    
    # Check input shape
    assert test_input.shape == (1, 1, 28, 28), f"Input shape is incorrect: {test_input.shape}"
    
    # Check output shape
    assert output.shape == (1, 10), f"Output shape is incorrect: {output.shape}"
    print(f"Model input shape: {test_input.shape}")
    print(f"Model output shape: {output.shape}")

def test_model_accuracy():
    # Load the latest model
    import glob
    import os
    
    model_files = glob.glob('models/*.pth')
    if not model_files:
        pytest.skip("No model file found")
    
    latest_model = max(model_files, key=os.path.getctime)
    model = MNISTModel()
    
    # Load state dict with error handling
    try:
        state_dict = torch.load(latest_model)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Retrain the model if loading fails
        from src.train import train
        train()
        # Try loading again
        latest_model = max(glob.glob('models/*.pth'), key=os.path.getctime)
        state_dict = torch.load(latest_model)
        model.load_state_dict(state_dict)
    
    # Test on validation set
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    BATCH_SIZE = 64
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 80, f"Model accuracy {accuracy:.2f}% is below 80%" 