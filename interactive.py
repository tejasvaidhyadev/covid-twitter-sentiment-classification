"Evaluate the model"""
import os
import json
import torch
import random
import logging
import argparse
import numpy as np
import util as util
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

from torch.utils.data import TensorDataset
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=23, help="random seed for initialization")
parser.add_argument('--trained_model', default="pretrained_model/finetuned_BERT_epoch_4.model")
parser.add_argument('--nolog', default=True, help="Logging of huggingface")
parser.add_argument('--pretrained_dir', default = "./pretrained_model", help="directory of pretrained library")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_label(preds, label_dict):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    pred_label= [label_dict_inverse[label] for label in preds_flat]

    return pred_label

def interAct(model, encoded_query, dataloader_tester, params_runtime , args,mark='Interactive', verbose=False):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()
    predictions = []
    for batch in dataloader_tester:
        
        batch = tuple(b.to(params_runtime.device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                 }
    
        outputs = model(**inputs)
        logits = outputs[0]
        #loss_val_total += loss.item()
        sm = nn.Softmax(dim=1)
        logits = sm(logits)
        
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
    predictions = np.concatenate(predictions, axis=0)
    
    with open(args.pretrained_dir+'/params.json', 'r') as fp:
        label_dict = json.load(fp)
    pred_label = get_label(predictions, label_dict)
    
    
    return(pred_label)


def bert_ner_init():
    args = parser.parse_args()
    
    if args.nolog:
        logging.disable(logging.INFO) # disable INFO and DEBUG logging everywhere
    
    Pre_trained_model_dir = args.pretrained_dir

    # Load the parameters from json file
    json_path = os.path.join(Pre_trained_model_dir, 'params_runtime.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params_runtime = util.Params(json_path)

    # Use GPUs if available
    params_runtime.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params_runtime.seed = args.seed

    # Set the logger
    util.set_logger(os.path.join(Pre_trained_model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    # Load the model
    tokenizer = BertTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2', 
                                          do_lower_case=True)

    model = BertForSequenceClassification.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2',
                                                      num_labels=3,
                                                      output_attentions=False,
                                                      output_hidden_states=False)
    print('loading pretraining weights')
    model.load_state_dict(torch.load(args.trained_model, map_location=torch.device('cpu') ))
    model.to(params_runtime.device)

    return model, params_runtime, tokenizer,args

def BertNerResponse(model, queryString):    
    model, params_runtime, tokenizer,args = model
    # tokenzing query
    
    with open('experiment/interactive/sentences.txt', 'w') as f:
        f.write(queryString)
    encoded_query = tokenizer.batch_encode_plus(
    [queryString,], 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
    )
    input_ids_test = encoded_query['input_ids']
    attention_masks_test = encoded_query['attention_mask']
    dataset_test = TensorDataset(input_ids_test, attention_masks_test)

    dataloader_tester = DataLoader(dataset_test, 
                                   sampler=SequentialSampler(dataset_test), 
                                   batch_size=1)
    result = interAct(model, encoded_query, dataloader_tester, params_runtime, args)
    return result


def main():

    model = bert_ner_init()
    print("======= ======= welcome! ======== ======= ")
    print("    1. Provide input strings")
    print("    2. Input 'exit' to end")
    while True:
        query = input('Input:')
        if query == 'exit':
            break
        print(BertNerResponse(model, query))


if __name__ == '__main__':
    main()
