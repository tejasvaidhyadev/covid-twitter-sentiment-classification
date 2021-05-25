import numpy as np 
import pandas as pd
import os
import glob
import pandas as pd
import sklearn
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from sklearn.metrics import f1_score
import random
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
#from utility import accuracy_per_class, f1_score_func
import argparse
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import json
from util import f1_score_func, accuracy_per_class
import util
import logging
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='digitalepidemiologylab/covid-twitter-bert', help="model name from huggingface")
parser.add_argument('--experiment_name', default='ct-BERT-2013setv1', help="model name from huggingface")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=8)


def evaluate(dataloader_val, model):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

if (__name__ == "__main__"):

    args = parser.parse_args()

    exp_name = args.experiment_name
    #assert Path("./"+exp_name).exists()
    os.mkdir(exp_name)

    util.set_logger(os.path.join(exp_name, 'train.log'))

    try:
        os.system("nvidia-smi")
    except:
        print("Something went wrong with nvidia-smi command")

    logging.info("loading all the files of Subtask SE data")

    filenames = [name for name in glob.glob('../2017_English_final/GOLD_without16/Subtask_A/twitter-20*')]
    df = pd.concat( [ pd.read_csv(f, sep='\t', names=['id', 'category','text']) for f in filenames ] )    
    df.set_index('id', inplace=True)
    logging.info("Loaded sucessfull")

    possible_labels = df.category.unique()
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    
    logging.info("label_dict: {}".format(label_dict))
    #os.path.join(exp_name, 'params.json')
    with open(os.path.join(exp_name, 'params.json'), 'w') as fp:
        json.dump(label_dict, fp)
    
    df['label'] = df.category.replace(label_dict)

    X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
                                                  df.label.values, 
                                                  test_size=0.15, 
                                                  random_state=17, 
                                                  stratify=df.label.values)

    df['data_type'] = ['not_set']*df.shape[0]
    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'val'


    # Data
    tokenizer = BertTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert', 
                                          do_lower_case=True)

    encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].text.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].text.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
    )


    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type=='train'].label.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_type=='val'].label.values)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)


 ## BERT MODEL
    model = BertForSequenceClassification.from_pretrained(args.model,
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

## DataLoader
    batch_size = args.batch_size
    logging.info("batch size: {}" .format(batch_size))
    dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)
    dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)

    optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)
    epochs = args.epochs
    logging.info("epochs: {}" .format(epochs))

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    logging.info("Device: {}" .format(device))
    #print(device)

    for epoch in tqdm(range(1, epochs+1)):
    
        model.train()
    
        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:

            model.zero_grad()
        
            batch = tuple(b.to(device) for b in batch)
        
            inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

            outputs = model(**inputs)
        
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
        
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
       
        torch.save(model.state_dict(), f'{exp_name}/finetuned_BERT_epoch_{epoch}.model')
        
        tqdm.write(f'\nEpoch {epoch}')
    
        loss_train_avg = loss_train_total/len(dataloader_train)            
        tqdm.write(f'Training loss: {loss_train_avg}')
    
        val_loss, predictions, true_vals = evaluate(dataloader_validation, model)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')
