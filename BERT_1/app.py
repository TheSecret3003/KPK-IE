from flask import Flask, request
from bert1 import get_predicted_candidate

app = Flask(__name__)

@app.route('/get-similar-entity-bert1/<version>', methods=['GET'])
def get_similar_entity_bert1(version):
    args = request.args
    nama_instansi = args.get('nama_instansi')
    nama_instansi = str(nama_instansi).lower()
    similar_entity = get_predicted_candidate(nama_instansi, reference_version=version)
    
    return similar_entity

if __name__ == '__main__':
    app.run(debug=True)
