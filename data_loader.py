import torch
from torch.utils.data import (DataLoader, TensorDataset)
import pandas as pd
from sklearn.model_selection import train_test_split

class MyDataLoader():
    def __init__(self, dataset_path, tokenizer, max_length, batch_size):
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.max_length = max_length
        self.batch_size = batch_size

    def load_csv(self, dataset_path):
        df = pd.read_csv(dataset_path)
        texts = df.text.values.tolist()
        labels = df.label.values.tolist()

        label_0 = [i for i in labels if int(i) == 0]
        label_1 = [i for i in labels if int(i) == 1]
        label_2 = [i for i in labels if int(i) == 2]

        print('has {} label 0 in {} total label'.format(len(label_0), len(labels)))
        print('has {} label 1 in {} total label'.format(len(label_1), len(labels)))
        print('has {} label 2 in {} total label'.format(len(label_2), len(labels)))

        train_x, val_x, train_y, val_y = train_test_split(texts, labels)

        return train_x, val_x, train_y, val_y
    
    def dataloader(self):
        train_x, val_x, train_y, val_y = self.load_csv(self.dataset_path)

        tokenizer_data_train = self.tokenizer.batch_encode_plus(train_x,
                                                        add_special_tokens=True,
                                                        return_attention_mask=True,
                                                        pad_to_max_length=True,
                                                        max_length=self.max_length,
                                                        truncation=True,
                                                        return_tensors='pt')
        tokenizer_data_val = self.tokenizer.batch_encode_plus(val_x,
                                                    add_special_tokens=True,
                                                    return_attention_mask=True,
                                                    pad_to_max_length=True,
                                                    max_length=self.max_length,
                                                    truncation=True,
                                                    return_tensors='pt')
        input_ids_train = tokenizer_data_train['input_ids']
        attention_masks_train = tokenizer_data_train['attention_mask']
        labels_train = torch.tensor(train_y)

        input_ids_val = tokenizer_data_val['input_ids']
        attention_masks_val = tokenizer_data_val['attention_mask']
        labels_val = torch.tensor(val_y)

        train_data = TensorDataset(
            input_ids_train, attention_masks_train, labels_train)
        val_data = TensorDataset(
            input_ids_val, attention_masks_val, labels_val)

        train_dataloader = DataLoader(
            train_data, batch_size= self.batch_size)
        
        val_dataloader = DataLoader(
            val_data, batch_size= self.batch_size)
        return train_dataloader, val_dataloader
                                                        