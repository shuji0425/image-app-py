import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 画像を Tensor に変換し、0~1の範囲に正規化する変換器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 平均と標準偏差で正規化
])

# CIFAR-10 の学習用データセット
train_detaset = datasets.CIFAR10(
    root="./data",      # 保存先のフォルダ
    train=True,         # 学習用データ
    download=True,      # データがなければダウンロード
    transform=transform # 前処理を適用
)

# データローダーを作成
train_loader = DataLoader(
    train_detaset, # 対象データセット
    batch_size=8,  # 一度に取り出す画像数
    shuffle=True   # 毎回順序をシャッフル
)

# バッチ1つを取り出して確認
data_iter = iter(train_loader)
images, labels = next(data_iter)

# バッチの形状を表示
print("画像の形状:", images.shape)
print("ラベルの形状:", labels.shape)
print("ラベル:", labels)