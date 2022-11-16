import search
import pandas as pd

def search_test(nama_instansi):
    # Read indexing table
    index_table = pd.read_csv('https://raw.githubusercontent.com/intanq/data/main/index_table.csv')
    # Read reference data
    reference_data = pd.read_csv('https://raw.githubusercontent.com/intanq/data/main/index_table.csv')

    search_ = search.Search(reference_data, index_table)
    phrase_candidates = search_.search(nama_instansi)
    return phrase_candidates


if __name__ == "__main__":
    refs = search_test("kementerian bumn")
    print(refs)