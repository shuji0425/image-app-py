from torchvision import transforms

# データセット
def get_default_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 平均と標準偏差で正規化
    ])