import numpy as np 
import pandas as pd
import os
import glob
import sklearn
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from sklearn.metrics import f1_score
import random
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
#from utility import accuracy_per_class, f1_score_func
from model import evaluate
import argparse
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import json
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--trained_model', default='./ct-BERTV2-13-14-15/finetuned_BERT_epoch_4.model', help="saved model name from huggingface")
parser.add_argument('--csv',help="csv of test sets")
def get_label(preds, label_dict):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    pred_label= [label_dict_inverse[label] for label in preds_flat]

    return pred_label



def evaluate2(dataloader_test, model):

    model.eval()
    
    loss_val_total = 0
    predictions, id_test = [], []
    
    progress_bar = tqdm(dataloader_test, desc='evaluating', leave=False, disable=False)
    for batch in progress_bar:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        #loss = outputs[0] //still calculating loss assuming labels ids
        logits = outputs[0]
        #loss_val_total += loss.item()
        sm = nn.Softmax(dim=1)
        logits = sm(logits)
        
        #logits = logits.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        ids = batch[2].cpu().numpy()
        predictions.append(logits)
        id_test.append(ids)

    predictions = np.concatenate(predictions, axis=0)
    id_test = np.concatenate(id_test, axis=0)
            
    return  predictions, id_test


if (__name__ == "__main__"):

    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert', 
                                          do_lower_case=True)
    df = pd.read_csv(args.csv, names=['id', 'category','text'])  
    df.set_index('id', inplace=True)
    
    encoded_data_test = tokenizer.batch_encode_plus(
    df.text.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
    )

    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(df.index.values)

    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

    model = BertForSequenceClassification.from_pretrained('digitalepidemiologylab/covid-twitter-bert',
                                                      num_labels=3,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

    model.load_state_dict(torch.load(args.trained_model))
    print("model loaded sucessfully")
    
    #device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with open('ct-BERTV2-13-14-15/params.json', 'r') as fp:
        label_dict = json.load(fp)
    
    print(label_dict)
    dataloader_tester = DataLoader(dataset_test, 
                                   sampler=SequentialSampler(dataset_test), 
                                   batch_size=8)
    
    
    predictions, id_test = evaluate2(dataloader_tester,model)
    pred_label = get_label(predictions, label_dict)
    id_test = id_test.flatten()
    

    filepath_out='results/'+args.csv.split('/')[-1]
    
    f =open(filepath_out,"w+")
    for i,j in zip(predictions, id_test):
        f.writelines('%s,%s\n' %(i, j))
    
    f.close()
    print("done one file")




    #accuracy_per_class(predictions, true_vals)


