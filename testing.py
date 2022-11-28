import pandas as pd
from torch import nn
from transformers import BertModel
import torch
import numpy as np
from transformers import BertTokenizer
import torch.nn.functional as F
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--range_awal", default=0, help="range awal")
parser.add_argument("--range_akhir", default=1000, help=" range akhir")
parser.add_argument("--model" , default=0, help="index model")
argss = vars(parser.parse_args())

test_df = pd.read_csv('DATA/v2/kandidat_table_ai.csv')
test_df = test_df.iloc[int(argss['range_awal']):int(argss['range_akhir']),:]

list_model = ['indobert_p1_models', 'indobert_p2_models']
INDEX = int(argss['model'])


tokenizer = BertTokenizer.from_pretrained('../../agus_pln/BASE_MODELS/{}'.format(list_model[INDEX]))
labels = {'no':0,
          'yes':1,
          }


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [labels[label] for label in df['status']]
        self.texts = [tokenizer(df['instansi'][i], df['sinonim'][i],
                               padding='max_length', max_length = 512, truncation=True, 
                               return_tensors="pt") for i in range(int(argss['range_awal']),int(argss['range_akhir']))]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('../../agus_pln/BASE_MODELS/{}'.format(list_model[INDEX]))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

model = torch.load('TORCH MODELS_{}'.format(list_model[INDEX]))

#Predict
def evaluate_with_indexing(model, test_data):

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    probs = []
    preds = []

    print("prediction start....")

    with torch.no_grad():

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
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


              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    new_test_data = test_data.copy()
    new_test_data['pred'] = list_pred
    new_test_data['prob'] = list_prob

    new_test_data = new_test_data[['instansi', 'sinonim', 'pred', 'prob']]

    return new_test_data
     
pred_result = evaluate_with_indexing(model, test_df)
pred_result.to_csv('result/indexing/hasil_prediksi_{}_{}.csv'.format(list_model[INDEX], argss['range_akhir']))