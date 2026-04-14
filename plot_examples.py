import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json

from src.dataset import create_dataloaders
from src.model import MLPClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
out_dir = Path("outputs/run_f_hyper_sgd")

# 1. Load Data (test_loader is deterministic with random_state=42)
data = create_dataloaders(
    data_dir=r"d:\DeepLearning\DL\csc4005-lab1-neu-mlp-tyanzuq2811\NEU-CLS_extracted",
    img_size=64,
    batch_size=32,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    augment=False,
    num_workers=0,
)

# 2. Load best model
model = MLPClassifier(input_dim=data.input_dim, num_classes=6, hidden_dims=[1024, 512, 128], dropout=0.3).to(device)
try:
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
except Exception as e:
    print(f"Error loading model: {e}")

model.eval()

# 3. Predict & gather examples
corrects = []
incorrects = []

with torch.no_grad():
    for x, y in data.test_loader:
        x_dev = x.to(device)
        logits = model(x_dev)
        preds = torch.argmax(logits, dim=1).cpu()
        
        for i in range(len(y)):
            img_tensor = x[i].view(64, 64)
            true_label = data.class_names[y[i].item()]
            pred_label = data.class_names[preds[i].item()]
            
            if preds[i] == y[i] and len(corrects) < 3:
                corrects.append((img_tensor, true_label, pred_label))
            elif preds[i] != y[i] and len(incorrects) < 3:
                incorrects.append((img_tensor, true_label, pred_label))
                
        if len(corrects) >= 3 and len(incorrects) >= 3:
            break

# 4. Plot
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("Ví dụ Dự đoán: Đúng (Trêm) và Sai (Dưới)", fontsize=16)

# Plot Corrects
for i, ax in enumerate(axes[0]):
    img, t, p = corrects[i]
    ax.imshow(img.numpy(), cmap='gray')
    ax.set_title(f"True: {t}\nPred: {p}", color='green')
    ax.axis("off")

# Plot Incorrects
for i, ax in enumerate(axes[1]):
    img, t, p = incorrects[i]
    ax.imshow(img.numpy(), cmap='gray')
    ax.set_title(f"True: {t}\nPred: {p}", color='red')
    ax.axis("off")

plt.tight_layout()
plt.savefig(out_dir / "predictions.png")
print(f"Đã lưu hình ảnh predictions.png vào {out_dir}")
