import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW,  get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Trainer():
    def __init__(self, model, tokenizer, args, train_loader, val_loader, device = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
    
    def train(self):
        learning_rate = self.args.lr
        epochs = self.args.epochs
        adam_epsilon = self.args.epsilon
        no_decay = ['bias', 'LayerNorm.weight']
        total_steps = len(self.train_loader) * epochs

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.2},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        best_f1_score = 0

        for epoch in tqdm(range(1, epochs+1)):
            preds, true_label = [], []
            loss_train_epochs = 0
            self.model.train()
            self.model.to(self.device)

            progress_bar = tqdm(self.train_loader, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
            for batch in progress_bar:
                self.model.zero_grad()

                batch_data = tuple(b.to(self.device) for b in batch)
                inputs = {'input_ids':      batch_data[0],
                        'attention_mask': batch_data[1],
                        'labels':         batch_data[2],
                        }
                outputs = self.model(**inputs)
                loss = outputs[0]
                loss_train_epochs += loss.item()
                pred = outputs[1].detach().cpu().numpy()
                pred = [np.argmax(lb).item() for lb in pred]
                true_label.append(inputs['labels'].cpu().numpy())
                preds.append(pred)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # loss_train_avg = loss_train_epochs/len(self.train_loader)
                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

            predictions = np.concatenate(preds, axis=0)
            true_train = np.concatenate(true_label, axis=0)

            accuracy_train = accuracy_score(true_train, predictions)
            # train_f1_train = f1_score(true_train, predictions, average="macro")
            # print("train accuracy on epochs:", epoch, "\naccuracy:", accuracy_train)
            # print("train f1_score on epochs:", epoch, "\nf1_score: ", train_f1_train)


            self.model.eval()
            loss_val_epochs = 0
            pred_vals, true_vals = [], []


            for batch in self.val_loader:
                batch = tuple(b.to(self.device) for b in batch)

                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'labels':         batch[2],
                        }
                with torch.no_grad():
                    outputs = self.model(**inputs)

                loss = outputs[0]
                loss_val_epochs += loss.item()

                pred = outputs[1].detach().cpu().numpy()
                pred = [np.argmax(lb).item() for lb in pred]
                true_vals.append(inputs['labels'].cpu().numpy())
                pred_vals.append(pred)
                
            pred_vals = np.concatenate(pred_vals, axis=0)
            true_vals = np.concatenate(true_vals, axis=0)

            accuracy_val = accuracy_score(true_vals, pred_vals)
            precision_val = precision_score(true_vals, pred_vals, average="macro")
            recall_val = recall_score(true_vals, pred_vals, average="macro")
            train_f1_val = f1_score(true_vals, pred_vals, average="macro")


            # print(train_f1_val)
            # print("val accuracy on epochs:", epoch, "\naccuracy:", accuracy_val)
            # print("val f1_score on epochs:", epoch, "\nf1_score: ", train_f1_val)

            if train_f1_val >=  best_f1_score:
                print("Accuracy score: ", accuracy_val)
                print("Precison score: ", precision_val)
                print("Recall score: ", recall_val)
                print("F1 score: ", train_f1_val)
                best_f1_score = train_f1_val
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                model_to_save.save_pretrained(self.args.model_save)
                self.tokenizer.save_pretrained(self.args.model_save)