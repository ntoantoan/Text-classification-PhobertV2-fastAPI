import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from utils import *
from typing import List
from arguments import load_args
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)
from torch.utils.data import (DataLoader, TensorDataset, Dataset)

app = FastAPI()


args = load_args()
MODEL_PATH = args.model_pretrained
MAX_LEN = args.max_length
BATCH_SIZE = args.batch_size
NUM_LABEL = 3
device = torch.device(0)


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, do_lower_case=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH,
                                                            num_labels=NUM_LABEL,
                                                            output_attentions=False,
                                                            output_hidden_states=False)
model = model.to(device)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id




class Loader_testDataset(Dataset):
    def __init__(self, list_id, list_text, tokenizer, max_len):
        self.list_text = list_text
        self.list_id = list_id
        self.max_len = max_len
        self.text = list_text
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.list_text)
    
    def __getitem__(self, index):
        text = self.text[index]
        idx = self.list_id[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            return_tensors='pt'
        )
        ids = inputs['input_ids'][0].to(device)
        mask = inputs['attention_mask'][0].to(device)
        
        return idx,{
            'input_ids': ids,
            'attention_mask': mask
        }


class text_sample(BaseModel):
    id: str
    text: str


class batch(BaseModel):
    list_item: List[text_sample]



@app.post("/predict-batch")
async def predict_batch(item: batch):
    input_data = item.list_item

    list_text = [data.text for data in input_data]
    list_text = [clean(text) for text in list_text]
    list_id = [data.id for data in input_data]
    id_preds = []
    label_preds = []
    prob_preds = []

    dataloader = Loader_testDataset(list_id, list_text, tokenizer, max_len=MAX_LEN)
    test_loader = DataLoader(dataloader, BATCH_SIZE, 
                         shuffle=False)
    

    model.eval()
    for idx, batch in test_loader:
        with torch.no_grad():
            outputs = model(**batch)
            preds_labels = torch.sigmoid(outputs[0])
            preds_labels = preds_labels.detach().cpu().numpy()
            preds = [np.argmax(lb).item() for lb in preds_labels]
            probs = [np.max(lb).item() for lb in preds_labels]
            id_preds.extend(idx)
            label_preds.extend(preds)
            prob_preds.extend(probs)
        

    list_id = [int(id) for id in id_preds]
    classification = {0: "tieu cuc", 1: "binh thuong", 2: "tich cuc"}
    probs = [str(round(i*100)) for i in prob_preds]
    labels = [classification[i] for i in label_preds if i in classification]

    outputs = [[{"id": list_id[i]}, {"label":labels[i]},{"probability": probs[i]}]  for i in range(len(list_id))]
    outputs = json.loads(json.dumps(outputs))
    return outputs

if __name__ == "__main__":
    uvicorn.run("app:app")

