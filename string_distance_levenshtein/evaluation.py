"""
Module to evaluate String Distance (Levenshtein Distance) model
"""

import pandas as pd
from get_similar_entity import get_similar_entity, get_similar_entity_norm, get_similar_entity_th
import ast


def get_similar_entity_1a(test_df, reference_version):
    """
    Function to automatically label test_df (distinct instansi from KPK's postgre table)
    """
    labeled_test_df = test_df.copy()
    candidates = []
    for index, row in labeled_test_df.iterrows():
        nama_inst = row[['nama_instansi']].values[0]
        print(f'processing index {index}: {nama_inst}')
        candidate = get_similar_entity(nama_inst, reference_version=reference_version)
        candidates.append(candidate)
    
    labeled_test_df['candidates'] = candidates
    return labeled_test_df

def get_similar_entity_1b(test_df, reference_version):
    """
    Function to automatically label test_df (distinct instansi from KPK's postgre table)
    """
    labeled_test_df = test_df.copy()
    candidates = []
    for index, row in labeled_test_df.iterrows():
        nama_inst = row[['nama_instansi']].values[0]
        print(f'processing index {index}: {nama_inst}')
        candidate = get_similar_entity_norm(nama_inst, reference_version=reference_version)
        candidates.append(candidate)
    
    labeled_test_df['candidates'] = candidates
    return labeled_test_df

def get_similar_entity_1c(test_df, reference_version):
    """
    Function to automatically label test_df (distinct instansi from KPK's postgre table)
    """
    labeled_test_df = test_df.copy()
    candidates = []
    for index, row in labeled_test_df.iterrows():
        nama_inst = row[['nama_instansi']].values[0]
        print(f'processing index {index}: {nama_inst}')
        candidate = get_similar_entity_th(nama_inst, reference_version=reference_version)
        candidates.append(candidate)
    
    labeled_test_df['candidates'] = candidates
    return labeled_test_df

def clean_test_df(test_df):
    """
    Function to clean test df
    Column to be cleaned: nama_instansi
    """
    cleaned = test_df.copy()
    cleaned['cleaned'] = cleaned['nama_instansi'].str.replace('/', ' ')
    cleaned['cleaned'] = cleaned['cleaned'].str.replace('[^\w\s]','',regex=True)
    cleaned['cleaned'] = cleaned['cleaned'].str.lower()
    cleaned = pd.DataFrame(cleaned['cleaned'].unique())
    cleaned = cleaned.rename(columns={0: 'nama_instansi'})
    cleaned = cleaned.reset_index()
    cleaned = cleaned.drop(['index'], axis=1)
    cleaned = cleaned.dropna()
    return cleaned

# def clean_labeled_data(labeled):
#     """
#     Function to clean labeled data (remove punctuations, transform to lowercase)
#     """
#     cleaned = labeled.copy()

#     # clean reference column
#     cleaned['cln reference'] = cleaned['reference'].str.replace('/', ' ')
#     cleaned['cln reference'] = cleaned['cln reference'].str.replace('[^\w\s]','',regex=True)
#     cleaned['cln reference'] = cleaned['cln reference'].str.lower()

#     # clean instansi column
#     cleaned['cln instansi'] = cleaned['instansi'].str.replace('/', ' ')
#     cleaned['cln instansi'] = cleaned['cln instansi'].str.replace('[^\w\s]','',regex=True)
#     cleaned['cln instansi'] = cleaned['cln instansi'].str.lower()

#     # drop reference and instansi columns
#     cleaned = cleaned.drop(['reference', 'instansi'], axis=1)

#     # rename cleaned columns
#     cleaned = cleaned.rename(columns={'cln reference': 'reference', 'cln instansi': 'instansi'})

#     return cleaned

# def get_true_references(cleaned_labeled_data, nama_instansi):
#     """
#     Function to get all true references from given nama_instansi
#     """
#     try:
#         official_instansi = cleaned_labeled_data[(cleaned_labeled_data['reference']==nama_instansi) & 
#                                                     (cleaned_labeled_data['status']=='yes')]['instansi'].values[0]
#     except:
#         official_instansi = cleaned_labeled_data[cleaned_labeled_data['instansi']==nama_instansi]['instansi'].values[0]
    
#     true_references = cleaned_labeled_data[(cleaned_labeled_data['instansi']==official_instansi)
#                         & (cleaned_labeled_data['status']=='yes')]['reference'].values
    
#     true_refs = []
#     for i in true_references:
#         true_refs.append(i)

#     true_refs.append(official_instansi)

#     return true_refs

# def predict_similar_entities(test_df, labeled_cleaned_df):
#     """
#     Function to predict similar entities using 
#     String Distance (Levenshtein Distance) model
#     """
#     eval_df = test_df.copy()
#     candidates_all = []
#     candidates_all_without_edit_dist = []
#     true_refs = []
    
#     for index, row in eval_df.iterrows():
#         nama_inst = row[['reference']].values[0]
#         candidates = get_candidates(nama_inst)
#         candidates_keys = [i for i in candidates.keys()]
#         true_ref = get_true_references(labeled_cleaned_df, nama_inst)
#         candidates_all.append(candidates)
#         candidates_all_without_edit_dist.append(candidates_keys)
#         true_refs.append(true_ref)
    
#     eval_df['candidates with edit dist'] = candidates_all
#     eval_df['candidates'] = candidates_all_without_edit_dist
#     eval_df['true candidates'] = true_refs

#     return eval_df

# def calc_accuracy_pct(pred_df):
#     """
#     Calculate accuracy prediction
#     """
#     pred_df_ = pred_df.copy()
#     checks = []
#     accuracy_pcts = []
#     for index, row in pred_df_.iterrows():
#         candidates = row[['candidates']].values[0]
#         true_candidates = row[['true candidates']].values[0]

#         if type(candidates) == str:
#             candidates = ast.literal_eval(candidates)
        
#         if type(true_candidates) == str:
#             true_candidates = ast.literal_eval(true_candidates)

#         check = []
#         for i in range(len(candidates)):
#             # print(i) # per character????
#             if candidates[i] in true_candidates:
#                 check.append(1)
#             else:
#                 check.append(0)
#         checks.append(check)
#         accuracy_pcts.append(sum(check)/len(check))

#     pred_df_['checks'] = checks
#     pred_df_['accuracy pcts'] = accuracy_pcts
#     return pred_df_[['reference', 'candidates with edit dist', 
#                     'candidates', 'true candidates', 'checks', 'accuracy pcts']]

