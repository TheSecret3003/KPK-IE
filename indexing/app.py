from flask import Flask, request
import pandas as pd
import search

"""
Example using Curl:
- curl "localhost:5000/get-references?nama_instansi=kementerian%20bumn"
"""

app = Flask(__name__)

@app.route('/get-references/<version>', methods=['GET'])
def get_references(version):
    # Read indexing table
    index_table = pd.read_csv(f'./data/indexing/{version}/index_table.csv')
    # Read reference data
    reference_data = pd.read_csv(f'./data/indexing/{version}/reference_data.csv')

    args = request.args
    nama_instansi = args.get('nama_instansi')

    search_ = search.Search(reference_data, index_table)
    phrase_candidates = search_.search(nama_instansi)
    return phrase_candidates

if __name__ == '__main__':
    app.run(debug=True)