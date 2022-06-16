from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchtext.data.functional import to_map_style_dataset

train = load_dataset("wmt16", "de-en",split=[f"train[:10%]"])

class customdataset(Dataset):
    def __init__(self,data):
        self.data = data
        print(len(self.data[0]))


    def __len__(self):
        return len(self.data[0])
        
    def __getitem__(self, idx):
        return self.data[0]['translation'][idx]['en'], self.data[0]['translation'][idx]['de']


train = customdataset(to_map_style_dataset(train))
for el in train:
    print(el)
