import torch
from homework.datasets.classification_dataset import load_data
from homework.models import Classifier, save_model

def train():
  train_data = load_data("./classification_data/train", shuffle=True, batch_size=256, num_workers=2)
  val_data = load_data("./classification_data/val", shuffle=False)

  model = Classifier()
  model = model.to("cuda")

  loss_func = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

  global_step = 0
  metrics = {"train_acc": [], "val_acc": []}
  num_epoch = 50
  for epoch in range(num_epoch):
    for key in metrics:
      metrics[key].clear()
    
    model.train()

    for img, label in train_data:
      img, label = img.to("cuda"), label.to("cuda")

      pred = model(img)
      loss_val = loss_func(pred, label)
      optimizer.zero_grad()
      loss_val.backward()
      optimizer.step()
      acc = (pred.argmax(dim=1) == label).float().mean().item()
      metrics["train_acc"].append(acc)

      global_step += 1

    with torch.inference_mode():
        model.eval()

        for img, label in val_data:
            img, label = img.to("cuda"), label.to("cuda")

            # TODO: compute validation accuracy
            pred = model(img)
            acc = (pred.argmax(dim=1) == label).float().mean().item()
            metrics["val_acc"].append(acc)

    # log average train and val accuracy to tensorboard
    epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
    epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

    # print on first, last, every 10th epoch
    if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_acc={epoch_train_acc:.4f} "
            f"val_acc={epoch_val_acc:.4f}"
        )

  save_model(model)

  # save a copy of model weights in the log directory
  torch.save(model.state_dict(), "classification.th")
  print(f"Model saved to {f'classification.th'}")

if __name__ == "__main__":
  train()
