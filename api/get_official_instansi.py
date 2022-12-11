import sys
sys.path.append('../')
import pandas as pd

def combine_splitted_data(reference_version='v2'):
    """
    Function to combine splitted data and return true pairs only
    """
    train = pd.read_csv(f'../indexing/data/splitted_data/{reference_version}/train.csv')
    val = pd.read_csv(f'../indexing/data/splitted_data/{reference_version}/val.csv')
    test = pd.read_csv(f'../indexing/data/splitted_data/{reference_version}/test.csv')

    combined = pd.concat([train, val, test])
    true_pairs = combined[combined['status']=='yes']
    true_pairs['cleaned instansi'] = true_pairs['instansi'].str.replace('/', ' ')
    true_pairs['cleaned instansi'] = true_pairs['cleaned instansi'].str.replace('[^\w\s]','',regex=True)
    true_pairs.reset_index(drop=True, inplace=True)
    return true_pairs

def get_official_instansi(true_pairs, returned_candidate, reference_version='v2'):
    """
    Function to get the official instansi name given returned_candidate
    Args:
    - returned_cadidate: str -> candidate returned from the model
    Returns:
    - str -> official instansi name from given returned_candidate
    """
    true_pairs = true_pairs
    try:
        return true_pairs[true_pairs['reference']==returned_candidate]['instansi'].iloc[0]
    except:
        pass

    try:
        return true_pairs[true_pairs['cleaned instansi']==returned_candidate]['instansi'].iloc[0]
    except:
        return 'Official Instansi Not Found'


# true_pairs = combine_splitted_data()
# print(get_official_instansi(true_pairs,'pt garuda indonesia persero tbk'))
# print(get_official_instansi(true_pairs,'bin'))
# print(get_official_instansi(true_pairs,'anri'))
# print(get_official_instansi(true_pairs,'itdc'))