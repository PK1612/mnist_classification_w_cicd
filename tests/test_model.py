import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MNISTModel
import torch
import pytest
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_parameters():
    model = MNISTModel()
    num_params = count_parameters(model)
    print(f"Total number of parameters: {num_params}")
    assert num_params < 25000, f"Model has {num_params} parameters, which exceeds the limit of 25000"

def test_input_output_shape():
    model = MNISTModel()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is incorrect: got {output.shape}, expected (1, 10)"

def test_model_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the latest model
    import glob
    import os
    
    model_files = glob.glob('models/*.pth')
    if not model_files:
        pytest.skip("No model file found")
    
    latest_model = max(model_files, key=os.path.getctime)
    model = MNISTModel().to(device)
    model.load_state_dict(torch.load(latest_model))
    
    # Test on validation set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Model accuracy: {accuracy:.2f}%")
    assert accuracy > 95, f"Model accuracy {accuracy:.2f}% is below 95%" 