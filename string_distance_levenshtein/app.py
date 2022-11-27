from flask import Flask, request
import pandas as pd
from get_candidates import get_candidates

app = Flask(__name__)

@app.route('/get-similar-entities', methods=['GET'])
def get_references():
    args = request.args
    nama_instansi = args.get('nama_instansi')
    similar_entities = get_candidates(nama_instansi)
    
    return similar_entities

if __name__ == '__main__':
    app.run(debug=True)