import sys 
sys.path.append('../')
from string_distance_levenshtein.search_utils import search
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer
import torch.nn.functional as F
from model import BertClassifier

#Model path
PATH = 'Models/indobert-base-p1'

#tokenizer and load fine-tuned model from path
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = torch.load(PATH)

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
def search_candidates(instansi) :  

    phrase_candidates = search(instansi,reference_version='v2')
    list_candidates = list(phrase_candidates.values())
    list_candidates= [x for y in list_candidates for x in y]

    if len(list_candidates)==0 :
        list_candidates.append('Bukan instansi BUMN, Kementerian, Pemerintah')

    candidates_df = pd.DataFrame()
    candidates_df['instance'] = [instansi]*len(list_candidates)
    candidates_df['candidate'] = list_candidates

    return(candidates_df)


#Predict function
def predict(model, test_data):

    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

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


    #get one candidate with highest score
    lst_sinonim = pred_result['candidate'].unique().tolist()

    idx_to_take = []
    for sinonim in lst_sinonim:
        df_sn = pred_result[pred_result['candidate']==sinonim]
        df_sn = df_sn.drop_duplicates()
        if 1 in df_sn['pred'].values:
            df_sn_new = df_sn[df_sn['pred']==1]
            idx = df_sn_new[df_sn_new['prob']==df_sn_new['prob'].max()].index.tolist()
            for id in idx :
                idx_to_take.append(id)
        elif 1 not in df_sn['pred'].values:
            df_sn_new = df_sn[df_sn['pred']==0]
            idx = df_sn_new[df_sn_new['prob']==df_sn_new['prob'].max()].index.tolist()
            if len(idx) > 1:
                print(len(idx))
            for id in idx :
                idx_to_take.append(id)

    new_result_df = pred_result.loc[idx_to_take,:]
    pred_candidate = new_result_df['candidate'][0]

    return pred_candidate


def get_predicted_candidate(instansi) :

    candidates_df = search_candidates(instansi=instansi)
    pred_result = predict(model, candidates_df)

    return pred_result


#testing
instansi = "Anggota DPRD Jawa Barat"
pred_candidate = get_predicted_candidate(instansi)

print(pred_candidate)
