from levenshtein import levenshtein
from search_utils import search

def get_similar_entity(nama_instansi, reference_version):
    """
    Function to get similar phrase candidates that have the lowest edit distance with
    respect to nama_instansi
    Args:
    - nama_instansi: str -> name of the instansi to be looked up
    Returns:
    - candidate with lowest edit distance
    """
    phrase_candidates = search(nama_instansi, reference_version)
    similar_phrases_dict = {}

    for word, candidates in phrase_candidates.items():
        for cand in candidates:
            # Define threshold
            threshold = 0
            if len(nama_instansi) < len(cand):
                threshold = len(cand) - len(nama_instansi)
            elif len(nama_instansi) > len(cand):
                threshold = len(nama_instansi) - len(cand)

            edit_dist = levenshtein(nama_instansi, cand)
            if edit_dist == 0:
                return cand
            elif edit_dist <= threshold:
                similar_phrases_dict[cand] = edit_dist        
    
    if len(similar_phrases_dict) != 0:
        return min(similar_phrases_dict, key=similar_phrases_dict.get)
    else:
        return 'Bukan instansi BUMN, Kementerian, Pemerintah'
    

def get_similar_entity_norm(nama_instansi, reference_version):
    """
    Function to get similar phrase candidates that have the lowest normalized edit distance with
    respect to nama_instansi
    Args:
    - nama_instansi: str -> name of the instansi to be looked up
    Returns:
    - candidate with lowest edit distance
    """
    phrase_candidates = search(nama_instansi, reference_version)
    similar_phrases_dict = {}

    for word, candidates in phrase_candidates.items():
        for cand in candidates:
            # Define threshold
            threshold = 0.5
            
            edit_dist = levenshtein(nama_instansi, cand)
            normalized_edit_dist = edit_dist/max(len(nama_instansi), len(cand))
            if normalized_edit_dist == 0:
                return cand
            elif normalized_edit_dist <= threshold:
                similar_phrases_dict[cand] = normalized_edit_dist

    sorted_ = dict(sorted(similar_phrases_dict.items(), key=lambda item: item[1], reverse=False))
    # return sorted_
        
    if len(sorted_) != 0:
        return min(sorted_, key=sorted_.get)
    else:
        return 'Bukan instansi BUMN, Kementerian, Pemerintah'

def get_similar_entity_norm_dep(nama_instansi, reference_version):
    """
    Function to get similar phrase candidates that have the lowest normalized edit distance with
    respect to nama_instansi
    Args:
    - nama_instansi: str -> name of the instansi to be looked up
    Returns:
    - candidate with lowest edit distance
    """
    phrase_candidates = search(nama_instansi, reference_version)
    similar_phrases_dict = {}

    for word, candidates in phrase_candidates.items():
        for cand in candidates:
            # Define threshold
            threshold = 0.5
            
            edit_dist = levenshtein(nama_instansi, cand)
            normalized_edit_dist = edit_dist/max(len(nama_instansi), len(cand))
            if normalized_edit_dist == 0:
                return cand
            elif normalized_edit_dist <= threshold:
                similar_phrases_dict[cand] = normalized_edit_dist

    sorted_ = dict(sorted(similar_phrases_dict.items(), key=lambda item: item[1], reverse=False))

    # Calc score
    score_threshold = 0.5
    scores = {}
    for key, val in sorted_.items():
        score = 0
        for word in key.split(' '):
            if word in nama_instansi:
                score += 1
        score_pct = score/len(nama_instansi.split(' '))
        if score_pct >= score_threshold:
            scores[key] = score_pct
    
    # return sorted_
        
    if len(scores) != 0:
        return scores
    else:
        return 'Bukan instansi BUMN, Kementerian, Pemerintah'

# print(get_similar_entity_norm('dinas pariwisata kabupaten gianyar', 'v2'))
# print(get_similar_entity_norm('dinas pariwisata provinsi riau', 'v2'))
# print(get_similar_entity_norm('pejabat bank negara indonesia', 'v2'))