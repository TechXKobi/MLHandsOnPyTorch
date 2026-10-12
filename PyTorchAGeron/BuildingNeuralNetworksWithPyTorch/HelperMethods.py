import torch 
import torchmetrics
import torch.nn as nn

class HelperML:

    def __init__(self, device):
        self.device = device

    def evaluate(self, model, metric, data_loader):
        model.eval()
        metric.reset()      # reset metric for each epoch

        with torch.no_grad():
            for *X_batch, y_batch in data_loader:
                X_batch = [X.to(self.device) for X in X_batch]
                y_batch = y_batch.to(self.device)
                y_pred = model(*X_batch)

                metric.update(y_pred, y_batch)         # update metric for each batch

        return metric.compute()

    def train(self, model, optimizer, criterion, metric, train_loader, val_loader, n_epochs, lr_scheduler=None, warmup_lr_scheduler=None): 
        history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}

        for epoch in range(n_epochs):
            if warmup_lr_scheduler is not None:
                warmup_lr_scheduler.step()
            total_loss = 0
            metric.reset()      # reset metric for each epoch

            for *X_batch, y_batch in train_loader:
                model.train()
                X_batch = [X.to(self.device) for X in X_batch]
                y_batch = y_batch.to(self.device)
                y_pred = model(*X_batch)

                loss = criterion(y_pred, y_batch)
                total_loss += loss.item()

                loss.backward()

                # Clip gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)

                optimizer.step()
                optimizer.zero_grad()
                metric.update(y_pred, y_batch)  # update metric for each batch

            mean_loss = total_loss / len(train_loader)

            history["train_losses"].append(mean_loss)
            history["train_metrics"].append(metric.compute().item())
            
            val_acc = self.evaluate(model, metric, val_loader).item()
            history["valid_metrics"].append(val_acc)

            print(f"Epoch {epoch+1}/{n_epochs},\n" 
                  + f"train_loss: {history["train_losses"][-1]:.4f},\n"
                  + f"train_metric: {history["train_metrics"][-1]:.4f},\n"
                  + f"valid_metric: {history["valid_metrics"][-1]:.4f}.\n")

            if lr_scheduler is not None:
                if warmup_lr_scheduler is not None:
                    if epoch >= 3:
                        lr_scheduler.step(val_acc)
                else:
                    lr_scheduler.step(val_acc)
                    
        return history

            