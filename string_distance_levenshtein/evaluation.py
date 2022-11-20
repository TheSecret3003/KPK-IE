"""
Module to evaluate String Distance (Levenshtein Distance) model
"""

import pandas as pd
from get_candidates import get_candidates
import ast

def clean_labeled_data(labeled):
    """
    Function to clean labeled data (remove punctuations, transform to lowercase)
    """
    cleaned = labeled.copy()

    # clean reference column
    cleaned['cln reference'] = cleaned['reference'].str.replace('/', ' ')
    cleaned['cln reference'] = cleaned['cln reference'].str.replace('[^\w\s]','',regex=True)
    cleaned['cln reference'] = cleaned['cln reference'].str.lower()

    # clean instansi column
    cleaned['cln instansi'] = cleaned['instansi'].str.replace('/', ' ')
    cleaned['cln instansi'] = cleaned['cln instansi'].str.replace('[^\w\s]','',regex=True)
    cleaned['cln instansi'] = cleaned['cln instansi'].str.lower()

    # drop reference and instansi columns
    cleaned = cleaned.drop(['reference', 'instansi'], axis=1)

    # rename cleaned columns
    cleaned = cleaned.rename(columns={'cln reference': 'reference', 'cln instansi': 'instansi'})

    return cleaned

def get_true_references(cleaned_labeled_data, nama_instansi):
    """
    Function to get all true references from given nama_instansi
    """
    try:
        official_instansi = cleaned_labeled_data[(cleaned_labeled_data['reference']==nama_instansi) & 
                                                    (cleaned_labeled_data['status']=='yes')]['instansi'].values[0]
    except:
        official_instansi = cleaned_labeled_data[cleaned_labeled_data['instansi']==nama_instansi]['instansi'].values[0]
    
    true_references = cleaned_labeled_data[(cleaned_labeled_data['instansi']==official_instansi)
                        & (cleaned_labeled_data['status']=='yes')]['reference'].values
    
    true_refs = []
    for i in true_references:
        true_refs.append(i)

    true_refs.append(official_instansi)

    return true_refs

def predict_similar_entities(test_df, labeled_cleaned_df):
    """
    Function to predict similar entities using 
    String Distance (Levenshtein Distance) model
    """
    eval_df = test_df.copy()
    candidates_all = []
    candidates_all_without_edit_dist = []
    true_refs = []
    
    for index, row in eval_df.iterrows():
        nama_inst = row[['reference']].values[0]
        candidates = get_candidates(nama_inst)
        candidates_keys = [i for i in candidates.keys()]
        true_ref = get_true_references(labeled_cleaned_df, nama_inst)
        candidates_all.append(candidates)
        candidates_all_without_edit_dist.append(candidates_keys)
        true_refs.append(true_ref)
    
    eval_df['candidates with edit dist'] = candidates_all
    eval_df['candidates'] = candidates_all_without_edit_dist
    eval_df['true candidates'] = true_refs

    return eval_df

def calc_accuracy_pct(pred_df):
    """
    Calculate accuracy prediction
    """
    pred_df_ = pred_df.copy()
    checks = []
    accuracy_pcts = []
    for index, row in pred_df_.iterrows():
        candidates = row[['candidates']].values[0]
        true_candidates = row[['true candidates']].values[0]

        if type(candidates) == str:
            candidates = ast.literal_eval(candidates)
        
        if type(true_candidates) == str:
            true_candidates = ast.literal_eval(true_candidates)

        check = []
        for i in range(len(candidates)):
            # print(i) # per character????
            if candidates[i] in true_candidates:
                check.append(1)
            else:
                check.append(0)
        checks.append(check)
        accuracy_pcts.append(sum(check)/len(check))

    pred_df_['checks'] = checks
    pred_df_['accuracy pcts'] = accuracy_pcts
    return pred_df_[['reference', 'candidates with edit dist', 
                    'candidates', 'true candidates', 'checks', 'accuracy pcts']]
