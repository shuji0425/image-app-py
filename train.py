import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader

# CNN 読み込み
from models.simple_cnn import SimpleCNN
from utils.dataset import get_default_transform

# 画像を Tensor に変換し、0~1の範囲に正規化する変換器
transform = get_default_transform()

# CIFAR-10 の学習用データセット
train_dataset = datasets.CIFAR10(
    root="./data",      # 保存先のフォルダ
    train=True,         # 学習用データ
    download=True,      # データがなければダウンロード
    transform=transform # 前処理を適用
)

# データローダーを作成
train_loader = DataLoader(
    train_dataset, # 対象データセット
    batch_size=8,  # 一度に取り出す画像数
    shuffle=True   # 毎回順序をシャッフル
)

# モデル、損失関数、最適化手法の設定
device = torch.device("cpu")
model = SimpleCNN().to(device)

#損失関数
criterion = nn.CrossEntropyLoss()
# Adamで学習率0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
num_epochs = 1
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in tqdm(enumerate(train_loader),
                                    total=len(train_loader),
                                    desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()             # 勾配を初期化
        outputs = model(inputs)           # 推論
        loss = criterion(outputs, labels) # 損失計算
        loss.backward()                   # 誤差逆伝播
        optimizer.step()                  # パラメータ更新

        running_loss += loss.item()
        if i % 100 == 99:
            print(f"[{epoch+1}, {i+1:5d}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

torch.save(model.state_dict(), "./outputs/simple_cnn.pth")
print("学習完了")