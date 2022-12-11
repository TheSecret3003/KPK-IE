import sys
sys.path.append('../')
from string_distance_levenshtein.search_utils import search
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer
import torch.nn.functional as F
from transformers import BertModel
from torch import nn
from  BERT_1.bert1_model import BertClassifier
from get_official_instansi import combine_splitted_data, get_official_instansi

# true pairs (pasangan nama resmi instansi dan sinonim)
true_pairs = combine_splitted_data('v2')

#load tokenizer
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')

#load fine-tuned model from path
PATH = '../BERT_1/Models/Best_Model/indobert-base-p1.pth'

#use cuda/GPU if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#Model
model = BertClassifier()
model.load_state_dict(torch.load(PATH))
model.to(device)
model.eval()


#Create Dataset
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        
        self.texts = [tokenizer(df['instance'][i], df['candidate'][i],
                               padding='max_length', max_length = 512, truncation=True, 
                               return_tensors="pt") for i in range(0,len(df))]

    
    def __len__(self):
        return len(self.texts)

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        return batch_texts


#find a list of candidates
def search_candidates(instansi, reference_version='v2') :  

    phrase_candidates = search(instansi,reference_version)
    list_candidates = list(phrase_candidates.values())
    list_candidates= [x for y in list_candidates for x in y]

    if len(list_candidates)==0 :
        list_candidates.append('Bukan instansi BUMN, Kementerian, Pemerintah')

    candidates_df = pd.DataFrame()
    candidates_df['instance'] = [instansi]*len(list_candidates)
    candidates_df['candidate'] = list_candidates

    return(candidates_df)


#Predict function
def predict(test_data):

    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    preds = []
    probs = []
    with torch.no_grad():

        for test_input in test_dataloader:

              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)
              prob = F.softmax(output, dim=1)
              pred = output.argmax(dim=1)
              probs.append(prob)
              preds.append(pred)

              #pred_class
              temp_pred = []
              for i in range(0,len(preds)):
                a = preds[i].cpu().numpy().tolist()
                temp_pred.append(a)
              list_pred = [x for y in temp_pred for x in y]

              #prob values
              temp_prob = []
              for i in range(0,len(probs)):
                a = probs[i].cpu().numpy().tolist()
                temp_prob.append(a)
              list_prob = [max(x) for y in temp_prob for x in y]
              
    pred_result = test_data.copy()
    pred_result['pred'] = list_pred
    pred_result['prob'] = list_prob

    #take the candidate value with the highest probability if pred == 1
    if 1 in pred_result['pred'].values:
        new_result_df= pred_result[pred_result['pred']==1]
        new_result_df = new_result_df.sort_values(by='prob', ascending=False)
        pred_candidate = new_result_df['candidate'].values[0]
        pred_candidate = get_official_instansi(true_pairs, pred_candidate)
    elif 1 not in pred_result['pred'].values:
        TRESHOLD = 0.6
        new_result_df= pred_result[pred_result['pred']==0]
        new_result_df = new_result_df.sort_values(by='prob', ascending=True)
        # If all predicted candidate values ​​= 0, then if the probability is smaller than the threshold --> the candidate will be taken
        if new_result_df['prob'][0] <= TRESHOLD :
            pred_candidate = new_result_df['candidate'].values[0]
            pred_candidate = get_official_instansi(true_pairs, pred_candidate)
        else :
            #If there is no prob value <= Treshold
            pred_candidate = "Bukan instansi BUMN, Kementerian, Pemerintah"
    
    return pred_candidate


def get_predicted_candidate(instansi, reference_version='v2') :
    instansi = instansi.lower()
    candidates_df = search_candidates(instansi=instansi, reference_version=reference_version)
    pred_result = predict(candidates_df)

    return pred_result

# #testing
# instansi = "Kementerian Kelautan Perikanan"
# pred_candidate = get_predicted_candidate(instansi,'v2')

# print(pred_candidate)
