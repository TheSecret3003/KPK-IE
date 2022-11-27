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