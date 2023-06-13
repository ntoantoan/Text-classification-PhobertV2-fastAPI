import os
import torch
from utils import *
from arguments import load_args
from data_loader import MyDataLoader
from train import Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

args = load_args()
data_path = args.data_path
pre_trained = args.model_pretrained
num_labels = args.num_class
max_length = args.max_length
batch_size = args.batch_size



tokenizer = AutoTokenizer.from_pretrained(pre_trained, do_lower_case=True)
model = AutoModelForSequenceClassification.from_pretrained(pre_trained,
                                                            num_labels=num_labels,
                                                            output_attentions=False,
                                                            output_hidden_states=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

dataloader = MyDataLoader(data_path, tokenizer, max_length, batch_size)

train_loader, val_loader = dataloader.dataloader()


trainer = Trainer(model, tokenizer, args, train_loader, val_loader, device)
trainer.train()


