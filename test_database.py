from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchtext.data.functional import to_map_style_dataset

train = load_dataset("wmt16", "de-en",split=f"train[:10%]")
print(train)
print(train.features)

class customdataset(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]['translation']['en'], self.data[idx]['translation']['de']

for el in customdataset(train):
    print(el)
# for el in train:
    # print(el['translation'])

# t = customdataset(train)

# for src, tgt in t:
    # print(src, tgt)
# def preprocess_function(examples):
    # d = {}
    # d['tuples'] = [(el['en'], el['de']) for el in examples['translation']]
    # return d
# train2 = train.map(preprocess_function, batched=True) 
# print(train2.features['tuples'])

# # train = customdataset(to_map_style_dataset(train))

# def preprocess(dataset):
    # for el in self.data[0]['translation']:
        # return el['en'], el['de']

# train_dataloader = DataLoader(
    # train.map(preprocess),
    # batch_size=32,
# )
# for el in train_dataloader:
    # print(el)
