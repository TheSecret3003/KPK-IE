from levenshtein import levenshtein
from search_utils import search
import itertools

def get_candidates(nama_instansi):
    """
    Function to get similar phrase candidates that have an edit distance 
    that is less than or equal passed edit distance threshold
    Args:
    - nama_instansi: str -> name of the instansi to be looked up
    Returns:
    - sorted_by_edit_dist: dict -> dictionary of resulted phrases and its edit distance
    """
    phrase_candidates = search(nama_instansi)
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
            if edit_dist <= threshold:
                similar_phrases_dict[cand] = edit_dist

    sorted_by_edit_dist = dict(sorted(similar_phrases_dict.items(), key=lambda item: item[1]))
    
    return sorted_by_edit_dist