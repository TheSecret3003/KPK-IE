import torch
import pandas as pd
from transformers import BertTokenizer, BertModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')

def sentence_to_vec(nama_instansi):
    """
    Function to transform given nama instansi to vector
    """
    tokenized_text = tokenizer.tokenize(nama_instansi)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    model = BertModel.from_pretrained('indobenchmark/indobert-base-p1',output_hidden_states = True)

    with torch.no_grad():
        outputs = model(tokens_tensor)
        hidden_states = outputs[2]

    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding