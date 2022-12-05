import sys
sys.path.append('../')
from flask import Flask, request
import torch
from bert1 import get_predicted_candidate
from BERT_1.model import BertClassifier
from bert2 import get_reference_cosine, get_reference_ffnn
from BERT_2b.ffnn import FeedforwardNeuralNetModel
from levenstein_model import get_similar_entity
from indexing.search import Search
import pandas as pd


"""
BERT 1
"""
#Model path
PATH_B1 = '../BERT_1/Models/indobert-base-p1'

#Load fine-tuned model from path
model = torch.load(PATH_B1)

"""
BERT 2b
"""
PATH_B2B = '../BERT_2b/Model/ffnn_best_model_v1.pt'
input_dim = 2304
hidden_dim = 500
output_dim = 2

device = torch.device('cpu')

model_b2b = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
model_b2b.load_state_dict(torch.load(PATH_B2B,  map_location=device))
model_b2b.eval()


"""
API
"""

app = Flask(__name__)

@app.route('/get-references/<version>', methods=['GET'])
def get_references(version):
    # Read indexing table
    index_table = pd.read_csv(f'../indexing/data/indexing/{version}/index_table.csv')
    # Read reference data
    reference_data = pd.read_csv(f'../indexing/data/indexing/{version}/reference_data.csv')

    args = request.args
    nama_instansi = args.get('nama_instansi')

    search_ = Search(reference_data, index_table)
    phrase_candidates = search_.search(nama_instansi)
    return phrase_candidates

@app.route('/get-similar-entity-bert1/<version>', methods=['GET'])
def get_similar_entity_bert1(version):
    args = request.args
    nama_instansi = args.get('nama_instansi')
    similar_entity = get_predicted_candidate(model, nama_instansi, reference_version=version)
    
    return list(similar_entity)[0]

@app.route('/get-similar-entity-bert2-cos/<version>', methods=['GET'])
def get_similar_entity_bert2_cos(version):
    args = request.args
    nama_instansi = args.get('nama_instansi')
    similar_entity = get_reference_cosine(nama_instansi, version)
    
    return similar_entity

@app.route('/get-similar-entity-bert2-ffnn/<version>', methods=['GET'])
def get_similar_entity_bert2_ffnn(version):
    args = request.args
    nama_instansi = args.get('nama_instansi')
    similar_entity = get_reference_ffnn(model_new=model_b2b, instansi=nama_instansi, reference_version=version)
    
    return similar_entity

@app.route('/get-similar-entity-lev/<version>', methods=['GET'])
def get_similar_entity_lev(version):
    args = request.args
    nama_instansi = args.get('nama_instansi')
    similar_entity = get_similar_entity(nama_instansi, reference_version=version)
    
    return similar_entity

if __name__ == '__main__':
    app.run(debug=True)