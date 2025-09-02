from model import MultiLayerPerceptronTorch
from torch import nn 
import torch

import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    
    # Create model, loss function and optimizer
    model = MultiLayerPerceptronTorch(bias=True).to(device=device)
    loss_fn = nn.MSELoss().to(device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    
    model.train()
    
    # Create training and testing data
    training_values = torch.linspace(-2*torch.pi, 2*torch.pi, 1000, dtype=torch.float32, device=device).view(-1, 100, 1)
    
    test_values = torch.rand(10, 100, 1, dtype=torch.float32, device=device) * 4 * torch.pi - 2 * torch.pi
    
    # Train the model
    for epoch in range(100000):
        total_train_loss = 0
        total_test_loss = 0
        
        # One epoch 
        for x in training_values:
            
            # Model's prediciton
            y = torch.sin(x)
            preds = model(x)
            loss = loss_fn(preds, y)
            total_train_loss += loss.mean()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Calculate total train loss
        total_train_loss /= training_values.shape[0]
        
        # Infer test data
        
        # Calculate total test loss
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Train Loss: {total_train_loss}, Test Loss: {total_test_loss}")
    
    
    # Visualize Result
    
    test_values_2 = torch.linspace(-2*torch.pi, 2*torch.pi, 1000, dtype=torch.float32, device=device).reshape(-1, 1)
    
    fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')
    ax.plot(test_values_2, torch.sin(test_values_2), label='sin(x)')
    ax.plot(test_values_2, model(test_values_2).detach().numpy(), label='MLP Approximation')
    plt.show()