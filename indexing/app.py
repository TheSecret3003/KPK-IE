from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import search

app = Flask(__name__)

@app.route('/get-references', methods=['GET'])
def get_references():
    # Read indexing table
    index_table = pd.read_csv('./data/v2/index_table.csv')
    # Read reference data
    reference_data = pd.read_csv('./data/v2/reference_data.csv')
    
    args = request.args
    nama_instansi = args.get('nama_instansi')

    search_ = search.Search(reference_data, index_table)
    phrase_candidates = search_.search(nama_instansi)
    return phrase_candidates

if __name__ == '__main__':
    app.run(debug=True)