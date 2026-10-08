"""BLOQUE 4: entrenamiento. Solo entrena; devuelve las pérdidas y no dibuja nada."""
import torch

from flow_matching import cfm_loss

def train(model, data, epochs=3000, batch_size=128, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    n_batches = (len(data) + batch_size - 1) // batch_size
    losses = []

    for epoch in range(epochs):
        order = torch.randperm(len(data), device='mps')           # barajar los datos en cada época
        total = 0.0
        for i in range(0, len(data), batch_size):
            batch = data[order[i:i + batch_size]]
            loss = cfm_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        losses.append(total / n_batches)
        if epoch % 100 == 0:
            print(f"época {epoch:4d}  pérdida {losses[-1]:.4f}")

    return losses
