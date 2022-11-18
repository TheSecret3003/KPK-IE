from levenshtein import levenshtein
from search_utils import search
import itertools

def get_candidates(nama_instansi, threshold=15):
    """
    Function to get similar phrase candidates that have an edit distance 
    that is less than or equal passed edit distance threshold
    Args:
    - nama_instansi: str -> name of the instansi to be looked up
    - threshold: int -> edit distance threshold
    Returns:
    - top5_phrases: dict -> dictionary of phrases and its edit distance (minimum)
    """
    phrase_candidates = search(nama_instansi)
    similar_phrases_dict = {}

    for word, candidates in phrase_candidates.items():
        for cand in candidates:
            edit_dist = levenshtein(nama_instansi, cand)
            if edit_dist <= threshold:
                similar_phrases_dict[cand] = edit_dist

    sorted_by_edit_dist = dict(sorted(similar_phrases_dict.items(), key=lambda item: item[1]))
    top5_phrases = dict(itertools.islice(sorted_by_edit_dist.items(), 5))
    
    return top5_phrases


# print(len(get_candidates('dewan ketahanan')))
# print(get_candidates('dewan ketahanan'))