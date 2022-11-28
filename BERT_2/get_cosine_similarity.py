from scipy.spatial.distance import cosine

def get_cosine_similarity(nama_instansi_vec, candidate_vec):
    """
    Function to get cosine similarity between given nama_instansi 
    and candidate (from search engine)
    """
    cosine_score = 1 - cosine(nama_instansi_vec, candidate_vec)
    return cosine_score