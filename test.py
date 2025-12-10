from datasets import SEEDDataset
from datasets import DEAPDataset
s = SEEDDataset(root_path="E:\\EEG_DataSets\\SEED\\Preprocessed_EEG\\")
print(s.get_dataset()["data"].shape)