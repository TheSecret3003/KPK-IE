from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import search

"""
raise HTTPError(req.full_url, code, msg, hdrs, fp)
urllib.error.HTTPError: HTTP Error 404: Not Found
"""

app = Flask(__name__)

@app.route('/get-references', methods=['GET'])
def get_references():
    # Read indexing table
    index_table = pd.read_csv('https://raw.githubusercontent.com/intanq/data/main/index_table.csv')
    # Read reference data
    reference_data = pd.read_csv('https://raw.githubusercontent.com/intanq/data/main/index_table.csv')

    nama_instansi = request.args.get('nama_instansi')

    search_ = search.Search(reference_data, index_table)
    phrase_candidates = search_.search(nama_instansi)
    return phrase_candidates

if __name__ == '__main__':
    app.run(debug=True)