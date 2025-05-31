import torch
from torchvision import datasets
import matplotlib.pyplot as plt
from models.simple_cnn import SimpleCNN
import matplotlib

from utils.dataset import get_default_transform

matplotlib.rcParams['font.family'] = 'AppleGothic'

# クラスレベル
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 前処理（trainと同じ）
transform = get_default_transform()

# テストデータセットを読み込む
# CIFAR-10 の学習用データセット
test_dataset = datasets.CIFAR10(
    root="./data",      # 保存先のフォルダ
    train=True,         # 学習用データ
    download=True,      # データがなければダウンロード
    transform=transform # 前処理を適用
)

# 1枚だけ取り出す
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=True
)

data_iter = iter(test_loader)
image, label = next(data_iter)

# モデル読み込み
device = torch.device("cpu")
model = SimpleCNN().to(device)
model.load_state_dict(
    torch.load("./outputs/simple_cnn.pth", map_location=device)
)
model.eval()

# 推論
with torch.no_grad():
    image = image.to(device)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

# 結果
print(f"正解: {classes[label.item()]}")
print(f"予測: {classes[predicted.item()]}")

# 可視化用
image = image.cpu()
image = image * 0.5 + 0.5
npimg = image.squeeze().numpy().transpose((1, 2, 0))

plt.imshow(npimg)
plt.title(f"予測: {classes[predicted.item()]}")
plt.show()