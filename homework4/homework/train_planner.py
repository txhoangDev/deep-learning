"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
from homework.datasets.road_dataset import load_data
from homework.models import save_model, load_model
from homework.metrics import PlannerMetric
import argparse
import torch
import numpy as np

def train(
  model_name: str = "mlp_planner",
  num_epoch: int = 50,
  lr: float = 1e-3,
  batch_size: int = 256,
  seed: int = 2024,
  num_workers: int = 2,
  **kwargs,
):
  torch.manual_seed(seed)
  np.random.seed(seed)
  train_data = load_data("./drive_data/train", shuffle=True, batch_size=batch_size, num_workers=num_workers)
  val_data = load_data("./drive_data/val", shuffle=False)

  device = "cuda"
  net = load_model(model_name, **kwargs)
  net = net.to(device)

  optim = torch.optim.AdamW(net.parameters(), lr=lr)
  criterion = torch.nn.L1Loss()

  planner_metric = PlannerMetric()

  for epoch in range(num_epoch):

    net.train()
    total_loss = 0
    for batch in train_data:
      waypoints = batch["waypoints"].to(device)
      if model_name == "cnn_planner":
        image = batch["image"].to(device)
        out = net(image)
      else:
        track_left = batch["track_left"].to(device)
        track_right = batch["track_right"].to(device)
        out = net(track_left, track_right)
      loss = criterion(out, waypoints)
      
      optim.zero_grad()
      loss.backward()
      optim.step()
      total_loss += loss.item()

    avg_train_loss = total_loss/len(train_data)

    planner_metric.reset()
    with torch.inference_mode():
      net.eval()

      total_long, total_lat = 0, 0
      n = 0
      for batch in val_data:
        waypoints_mask = batch["waypoints_mask"].to(device)
        waypoints = batch["waypoints"].to(device)
        if model_name == "cnn_planner":
          image = batch["image"].to(device)
          out = net(image)
        else:
          track_left = batch["track_left"].to(device)
          track_right = batch["track_right"].to(device)
          out = net(track_left, track_right)
      
        planner_metric.add(out, waypoints, waypoints_mask)
      
      results = planner_metric.compute()
      if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
        print(
                f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} "
                f"| Longitudinal: {results['longitudinal_error']:.4f} "
                f"| Lateral: {results['lateral_error']:.4f} "
                f"| L1: {results['l1_error']:.4f}"
            )
        save_model(net)
        torch.save(net.state_dict(), f"{model_name}_{epoch}.th")
        print(f"Model saved to {f'{model_name}_{epoch}.th'}")

  save_model(net)
  torch.save(net.state_dict(), f"{model_name}.th")
  print(f"Model saved to {f'{model_name}.th'}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--model_name", type=str, required=True)
  parser.add_argument("--num_epoch", type=int, default=50)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--seed", type=int, default=2024)
  parser.add_argument("--num_workers", type=int, default=2)
  parser.add_argument("--batch_size", type=int, default=256)
  train(**vars(parser.parse_args()))
