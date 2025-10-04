from homework.datasets.road_dataset import load_data
from homework.models import Detector, save_model
from homework.metrics import ConfusionMatrix
import torch
import torchvision

def train():
  train_data = load_data("./drive_data/train", shuffle=True, batch_size=16, num_workers=2)
  val_data = load_data("./drive_data/val", shuffle=False)

  if torch.cuda.is_available():
    device = torch.device("cuda")
  elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
  else:
    print("CUDA not available, using CPU")
    device = torch.device("cpu")
  model = Detector()
  model.to(device)

  loss_func = torch.nn.CrossEntropyLoss()
  depth_loss_func = torch.nn.L1Loss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

  global_step = 0
  metrics = {"train_acc": [], "val_acc": []}
  num_epoch = 50
  for epoch in range(num_epoch):
      for key in metrics:
          metrics[key].clear()

      model.train()
      train_cm = ConfusionMatrix(num_classes=3)
      train_mae = []
      for batch in train_data:
        images = batch["image"].to(device)
        seg_targets = batch["track"].to(device)
        depth_targets = batch["depth"].to(device)

        if torch.rand(1) < 0.5:
          images = torch.flip(images, dims=[3])
          seg_targets = torch.flip(seg_targets, dims=[2])
          depth_targets = torch.flip(depth_targets, dims=[2])

        seg_logits, depth_pred = model(images)
        seg_loss = loss_func(seg_logits, seg_targets)
        reg_loss = depth_loss_func(depth_pred, depth_targets)
        loss = seg_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_seg = seg_logits.argmax(dim=1)
        seg_acc = (pred_seg == seg_targets).float().mean().item()
        metrics["train_acc"].append(seg_acc)

        train_cm.add(pred_seg, seg_targets)
        train_mae.append((depth_pred - depth_targets).abs().mean().item())

        global_step += 1

      train_miou = train_cm.compute()
      train_mae_val = sum(train_mae) / len(train_mae)
      
      with torch.inference_mode():
          model.eval()

          val_cm = ConfusionMatrix(num_classes=3)
          val_mae = []

          for batch in val_data:
            images = batch["image"].to(device)
            seg_targets = batch["track"].to(device)
            depth_targets = batch["depth"].to(device)

            seg_logits, depth_pred = model(images)
            pred_seg = seg_logits.argmax(dim=1)
            seg_acc = (pred_seg == seg_targets).float().mean().item()
            metrics["val_acc"].append(seg_acc)

            val_cm.add(pred_seg, seg_targets)
            val_mae.append((depth_pred - depth_targets).abs().mean().item())

      val_miou = val_cm.compute()
      val_mae_val = sum(val_mae) / len(val_mae)

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
          print(f"train_mIoU={train_miou}, val_mIoU={val_miou}, train_MAE={train_mae_val:.4f}, val_MAE={val_mae_val:.4f}")

  save_model(model)

  # save a copy of model weights in the log directory
  torch.save(model.state_dict(), "detector.th")
  print(f"Model saved to {f'detector.th'}")

if __name__ == "__main__":
  train()