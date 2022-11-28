from levenshtein import levenshtein
from search_utils import search

# Algorithm 1a
def get_similar_entity(nama_instansi, reference_version='v2'):
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
    
# Algorithm 1b
def get_similar_entity_norm(nama_instansi, reference_version='v2'):
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
    
# Algorithm 1c
def get_similar_entity_th(nama_instansi, reference_version='v2'):
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
            max_len = max(len(nama_instansi), len(cand))
            edit_dist = levenshtein(nama_instansi, cand)

            if edit_dist == 0:
                return cand
            elif edit_dist <= max_len*threshold:
                similar_phrases_dict[cand] = edit_dist, max_len*threshold

    sorted_ = dict(sorted(similar_phrases_dict.items(), key=lambda item: item[1], reverse=False))
        
    if len(sorted_) != 0:
        return min(sorted_, key=sorted_.get)
    else:
        return 'Bukan instansi BUMN, Kementerian, Pemerintah'
