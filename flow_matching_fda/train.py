"""BLOCK 4: training. It only trains and returns losses.

The data comes as an FDataGrid and is converted once into a torch tensor [N, M]
on the model's device, because the net and the CFM loss work with torch.
"""
import torch
from flow_matching_fda.data import to_tensor
from flow_matching_fda.flow_matching import cfm_loss


def train(model, data, epochs=3000, batch_size=128, lr=1e-4):
    device = next(model.parameters()).device
    x = to_tensor(data, device)                                 # FDataGrid -> [N, M]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    n_batches = (len(x) + batch_size - 1) // batch_size
    losses = []

    for epoch in range(epochs):
        order = torch.randperm(len(x), device=device)
        total = 0.0
        for i in range(0, len(x), batch_size):
            batch = x[order[i:i + batch_size]]
            loss = cfm_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        losses.append(total / n_batches)
        if epoch % 100 == 0:
            print(f"epoch {epoch:4d}  loss {losses[-1]:.4f}")
    return losses
