import pandas as pd
import sys
sys.path.append('../')
import indexing.search

def search (nama_instansi, reference_version):
    reference_data = pd.read_csv(f'../indexing/data/indexing/{reference_version}/reference_data.csv')
    index_table = pd.read_csv(f'../indexing/data/indexing/{reference_version}/index_table.csv')

    s = indexing.search.Search(reference_data, index_table)

    phrase_candidates = s.search(nama_instansi)
    return phrase_candidates