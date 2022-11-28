from sentence_to_vec import sentence_to_vec
from get_cosine_similarity import get_cosine_similarity
from search_utils import search

def get_synonim(nama_instansi, reference_version='v2'):
    """
    Function to get the synonym of given nama_instansi
    Args:
    - nama_instansi: inputted nama instansi
    - reference_version: indexing version (latest = v2)
    """
    phrase_candidates = search(nama_instansi, reference_version)
    similar_phrases_dict = {}
    print('vectorise nama_instansi...')
    nama_instansi_vec = sentence_to_vec(nama_instansi)

    for word, candidates in phrase_candidates.items():
        for cand in candidates:
            threshold = 0.5
            print('vectorise candidate...')
            candidate_vec = sentence_to_vec(cand)
            cosine_score = get_cosine_similarity(nama_instansi_vec, candidate_vec)
            if cosine_score >= threshold:
                similar_phrases_dict[cand] = cosine_score

    if len(similar_phrases_dict) != 0:
        return max(similar_phrases_dict, key=similar_phrases_dict.get)
    else:
        return 'Bukan instansi BUMN, Kementerian, Pemerintah'

# print(get_synonim('pejabat pelni', reference_version='v2'))
