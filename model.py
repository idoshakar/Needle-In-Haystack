import torch
import torch.nn as nn
import pandas as pd

import process_recipe_cuisine

class Cuisine(nn.Module):
    def __init__(self, x_len: int, y_len: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(x_len, 128),
            nn.ReLU(),
            nn.Linear(128, y_len)
        )

    def forward(self, x):
           return self.fc(x)

def main():
# Load a simple JSON file
    with process_recipe_cuisine.get_database() as con:
        x_len = len(process_recipe_cuisine.get_ingredient_names(con))
        y_len = len(process_recipe_cuisine.get_cuisine_names(con))
    dataloader = lambda: pd.read_csv('output.csv', chunksize = 2**12)
    model = Cuisine(x_len, y_len)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 10
    last_batch_X, last_batch_y = None, None
    for epoch in range(num_epochs):
        last_batch_X, last_batch_y = None, None
        for batch in dataloader():
            features = batch.drop(columns=["id"]).iloc[:, :x_len].astype(float).values
            labels   = batch.drop(columns=["id"]).iloc[:, x_len:].astype(int).values

            batch_X = torch.tensor(features, dtype=torch.float32)

            # if you have *one-hot* cuisine labels
            batch_y = torch.tensor(labels, dtype=torch.float32)

            if last_batch_X is not None:
                # 1. Forward Pass: Calculate predictions
                outputs = model(last_batch_X)
                print(torch.tensor(outputs.sum(axis=1) > 1.0, dtype=torch.long))

                # 2. Calculate Loss: Compare predictions to true labels
                loss = loss_fn(outputs, last_batch_y)

                # 3. Backpropagation: Calculate gradients
                optimizer.zero_grad()
                loss.backward()

                # 4. Update Weights: Take a step with the optimizer
                optimizer.step()

                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

            last_batch_X = batch_X
            last_batch_y = batch_y

    model.eval() # Set the model to evaluation mode
    test_loss = 0.0
    with torch.no_grad(): # Disable gradient calculation
        print("\nStarting evaluation on test set...")
        if last_batch_X is not None:
            outputs = model(last_batch_X)
            loss = loss_fn(outputs, last_batch_y)
            test_loss += loss.item()

    avg_test_loss = test_loss
    print(f'Average Test Loss: {avg_test_loss:.4f}')

if __name__ == '__main__':
    main()
