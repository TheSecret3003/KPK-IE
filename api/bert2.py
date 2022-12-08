import pandas as pd
import sys
sys.path.append('../')
from search_utils import search
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import torch
from BERT_2.ffnn import FeedforwardNeuralNetModel

tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = BertModel.from_pretrained('indobenchmark/indobert-base-p1',output_hidden_states = True)

def sentence_to_vec(sentence):
  tokenized_text = tokenizer.tokenize(sentence)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  tokens_tensor = torch.tensor([indexed_tokens])

  with torch.no_grad():
    outputs = model(tokens_tensor)
    hidden_states = outputs[2]

  token_vecs = hidden_states[-2][0]
  sentence_embedding = torch.mean(token_vecs, dim=0)
  return sentence_embedding

def get_reference_cosine(instansi, reference_version='v2'):
  references = search(instansi,reference_version)
  score = 0
  temp_candidate = ""
  instansi_vec = sentence_to_vec(instansi)
  for word, candidates in references.items():
    for candidate in candidates:
      reference_vec = sentence_to_vec(candidate)
      same = 1- cosine(instansi_vec, reference_vec)
      if same >= score:
        score = same
        temp_candidate = candidate
    
  if temp_candidate == "":
    temp_candidate = 'Bukan instansi BUMN, Kementerian, Pemerintah'
  elif score<=0.8:
    temp_candidate = 'Bukan instansi BUMN, Kementerian, Pemerintah'
  return temp_candidate


def get_reference_ffnn(model_new, instansi, reference_version='v2'):
  references = search(instansi,reference_version)
  # print(references)
  score = 0
  temp_candidate = ""
  instansi_vec = sentence_to_vec(instansi)
  for word, candidates in references.items():
    for candidate in candidates:
      reference_vec = sentence_to_vec(candidate)
      diff_vec = torch.abs(torch.sub(instansi_vec, reference_vec))
      merge = torch.cat((instansi_vec,reference_vec,diff_vec))
      probs = model_new(merge)
      temp_score = probs[1].item()
      if temp_score >= score:
        score = temp_score
        temp_candidate = candidate
    
  if temp_candidate == "":
    temp_candidate = 'Bukan instansi BUMN, Kementerian, Pemerintah'
  elif score<=0.5:
    temp_candidate = 'Bukan instansi BUMN, Kementerian, Pemerintah'
  return temp_candidate
