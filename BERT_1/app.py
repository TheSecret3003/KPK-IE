from flask import Flask, request
from bert1 import get_predicted_candidate
from model import BertClassifier
import torch

#Model path
PATH = './Models/Best_Model/indobert-base-p1'

#Load fine-tuned model from path
model = torch.load(PATH)

app = Flask(__name__)

@app.route('/get-similar-entity-bert1/<version>', methods=['GET'])
def get_similar_entity_bert1(version):
    args = request.args
    nama_instansi = args.get('nama_instansi')
    similar_entity = get_predicted_candidate(model, nama_instansi, reference_version=version)
    
    return list(similar_entity)[0]

if __name__ == '__main__':
    app.run(debug=True)
