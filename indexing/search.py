import ast

class Search(object):
    """
    Class Search to perform search
    """
    def __init__(self, reference_data, index_table):
        self.reference_data = reference_data
        self.index_table = index_table
    
    def search(self, nama_instansi):
        """
        Function to search candidates from reference data given nama_instansi
        Args:
        - nama_instansi: str -> name of instansi
        Returns:
        - phrase_candidates: dict -> dictionary of each word in given nama_instansi with value of its occurence in reference data
        """
        words = nama_instansi.split()
        index_candidates = {}
        phrase_candidates = {}

        for word in words:
            try:
                indexes = self.index_table[self.index_table['Kata'].str.lower()==word.lower()]['Index'].iloc[0]
                if type(indexes) == str:
                    indexes = ast.literal_eval(indexes)
                index_candidates[word] = indexes
            except:
                # word is not found in Index table
                continue

        for word, indexes in index_candidates.items():
            words = []
            for i in indexes:
                words.append(self.reference_data.iloc[int(i)].iloc[1])
            phrase_candidates[word] =  words
        return phrase_candidates