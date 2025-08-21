import torch, torchvision.transforms as T
from PIL import Image
from src.train import FusionModel
import pandas as pd
def predict(image_path, tab_values):
    model = FusionModel(tab_dim=len(tab_values))
    model.load_state_dict(torch.load('./models/task3_fusion.pth', map_location='cpu'))
    model.eval()
    transform = T.Compose([T.Resize((224,224)), T.ToTensor()])
    img = transform(Image.open(image_path).convert('RGB')).unsqueeze(0)
    tab = torch.tensor([tab_values]).float()
    with torch.no_grad():
        return model(img, tab).item()
if __name__=='__main__':
    print(predict('./data/task3_images/sample.jpg',[1200,3]))
