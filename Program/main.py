import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 5)  # 3 input features, 5 output features
        self.fc2 = nn.Linear(5, 2)  # 5 input features, 2 output features

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create some dummy data
input_data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
target_data = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # Target data for the two samples

# Initialize the model
model = SimpleNet()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    output = model(input_data)

    # Compute the loss
    loss = criterion(output, target_data)

    # Backward pass
    loss.backward()

    # Update the weights
    optimizer.step()

    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the trained model
test_data = torch.tensor([[7.0, 8.0, 9.0]])
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predicted_output = model(test_data)
    print("Predicted Output:")
    print(predicted_output)
