import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import pandas as pd
from PIL import Image
import os

class HouseDataset(Dataset):
    def __init__(self, df, img_dir, tab_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tab_cols = tab_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        tab = torch.tensor(row[self.tab_cols].values.astype('float32'))
        price = torch.tensor(row['price']).float()
        return img, tab, price


class FusionModel(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()
        base = models.resnet18(pretrained=True)
        modules = list(base.children())[:-1]  # remove final FC
        self.cnn = nn.Sequential(*modules)
        self.tab = nn.Sequential(
            nn.Linear(tab_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.head = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, img, tab):
        img_feat = self.cnn(img).view(img.size(0), -1)  # (B, 512)
        tab_feat = self.tab(tab)
        x = torch.cat([img_feat, tab_feat], dim=1)
        return self.head(x).squeeze(1)


def main():
    # placeholder dataset (requires sample.jpg to exist!)
    df = pd.DataFrame({
        'image': ['sample.jpg', 'sample.jpg'],
        'size': [1200, 1500],
        'beds': [3, 4],
        'price': [250000, 300000]
    })

    img_dir = './data/task3_images'
    os.makedirs(img_dir, exist_ok=True)

    tab_cols = ['size', 'beds']
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    ds = HouseDataset(df, img_dir, tab_cols, transform)
    dl = DataLoader(ds, batch_size=2)

    model = FusionModel(tab_dim=len(tab_cols))

    for img, tab, price in dl:
        pred = model(img, tab)
        print('Prediction shape:', pred.shape)
        print('Pred values:', pred)

    # ✅ Ensure models/ folder exists
    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), './models/task3_fusion.pth')


if __name__ == '__main__':
    main()
