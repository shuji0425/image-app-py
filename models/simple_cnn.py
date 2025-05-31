import torch.nn as nn
import torch.nn.functional as F

# nn.Moduleを継承して自作モデルを定義
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 畳み込み層1: 入力3チャネル(RGB), 出力16チャネル, カーネルサイズ3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # 畳み込み層2: 入力チャネル16, 出力32
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 畳み込み層3: 入力チャネル32, 出力64
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # 全結合層1: 64チャネル×4×4 -> 128ユニット
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        # 全結合層2: 128ユニット -> 10クラス
        self.fc2 = nn.Linear(128, 10)

        # プーリング層
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # 入力: [batch, 3, 32, 32]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten: [batch, 64, 4, 4] → [batch, 64*4*4]
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x)) # ユニット128
        x = self.fc2(x)         # クラス10
        return x