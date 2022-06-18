from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader, Dataset
from torchtext.data.functional import to_map_style_dataset

# train = load_dataset("wmt16", "de-en",split=f"train[:10%]")
# print(train)
# print(train.features)

class customdataset(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]['en'], self.data[idx]['pt']

# # for el in customdataset(train):
    # # print(el)
# train, val, test = load_dataset("wmt16", "de-en",split=[f"train[:1%]","validation[:1%]","test[:1%]"])

# print(train)
# train, val, test = load_dataset("wmt16", "de-en",split=["train[:1%]","validation[:1%]","test[:1%]"])

# all_ds2 = load_dataset("wmt16", "de-en",split="train[:1%]+validation[:1%]+test[:1%]")

train, val, test = load_dataset("csv", data_files=["data/out.csv"], split=['train[:95%]','train[95%:97%]','train[97%:]'])

print(train, val, test)



# all_ds2.save_to_disk("trainvaltest.csv")
# dataset_dict.save_to_disk("../data/wikipedia_rank_nocache")

# all_ds = concatenate_datasets([train, val, test])
# print(all_ds)
# print(all_ds2)
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

# train = customdataset(to_map_style_dataset(train))

# def preprocess(dataset):
    # for el in self.data[0]['translation']:
        # return el['en'], el['de']

train_dataloader = DataLoader(
        customdataset(train)
)
for el in train_dataloader:
    print(type(el[0]))
