from BERT_2b import get_reference_cosine, get_reference_ffnn
from flask import Flask, request

app = Flask(__name__)

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
    similar_entity = get_reference_ffnn(nama_instansi, version)
    
    return similar_entity

if __name__ == '__main__':
    app.run(debug=True)

