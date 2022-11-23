from flask import Flask, request
import pandas as pd
from get_similar_entity import get_similar_entity

app = Flask(__name__)

@app.route('/get-similar-entities/<version>', methods=['GET'])
def get_references(version):
    args = request.args
    nama_instansi = args.get('nama_instansi')
    similar_entities = get_similar_entity(nama_instansi, reference_version=version)
    
    return similar_entities

if __name__ == '__main__':
    app.run(debug=True)