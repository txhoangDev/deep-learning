"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
from homework.datasets.road_dataset import load_data
from homework.models import MLPPlanner, save_model
from homework.metrics import PlannerMetric
import torch

train_data = load_data("./drive_data/train", shuffle=True, batch_size=256, num_workers=2)
val_data = load_data("./drive_data/val", shuffle=False)

device = "cuda"
net = MLPPlanner()
net = net.to(device)

optim = torch.optim.AdamW(net.parameters(), lr=1e-3)
criterion = torch.nn.L1Loss()

planner_metric = PlannerMetric()

num_epoch = 50
for epoch in range(num_epoch):

  net.train()
  total_loss = 0
  for batch in train_data:
    waypoints = batch["waypoints"].to(device)
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
      torch.save(net.state_dict(), "mlp_planner_{epoch}.th")
      print(f"Model saved to {f'mlp_planner_{epoch}.th'}")

save_model(net)
torch.save(net.state_dict(), "mlp_planner.th")
print(f"Model saved to {f'mlp_planner.th'}")
