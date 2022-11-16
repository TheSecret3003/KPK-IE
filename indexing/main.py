import pandas as pd
import search
import preprocess_reference_data
import index

def create_reference_data(version):
    prd = preprocess_reference_data.PreprocessReferenceData()
    
    train = pd.read_csv(f'./data/splitted_data/{version}/train.csv')
    val = pd.read_csv(f'./data/splitted_data/{version}/val.csv')
    test = pd.read_csv(f'./data/splitted_data/{version}/test.csv')
    labeled_df = pd.concat([train,val,test])

    bumn_df = pd.read_csv('./data/from_kpk/Daftar BUMN AI.csv')
    klpd_df = pd.read_csv('./data/from_kpk/Daftar KLPD PPG AI.csv')

    reference_data = prd.combine_reference_data(labeled_df, bumn_df, klpd_df)
    reference_data.to_csv(f'./data/indexing_{version}/reference_data.csv')
    return reference_data

def create_index_table(version, reference_data):
    idx = index.Index(reference_data)

    index_table = idx.get_index_table()
    index_table.to_csv(f'./data/indexing_{version}/index_table.csv')

    return index_table


if __name__ == "__main__":
    # reference_data = create_reference_data('v1')
    # print(ref)

    reference_data = pd.read_csv('./data/indexing_v1/reference_data.csv')

    # index_table = create_index_table('v1', reference_data)
    # print(index_table)

    index_table = pd.read_csv('./data/indexing_v1/index_table.csv')

    # Test search
    s = search.Search(reference_data, index_table)

    nama_instansi = "dewan ketahanan nasional"
    phrase_candidates = s.search(nama_instansi)
    print(phrase_candidates)



