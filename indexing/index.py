import pandas as pd

class Index(object):
    """
    Class Index to perform indexing
    """
    def __init__(self, reference_data):
        self.reference_data = reference_data
        self.unique_words = self.get_unique_words()
        self.indexes = self.find_indexes()
        self.index_table = self.create_index_table()

    def find_indexes(self):
        """
        Function to search row(s) in reference_data that contains each unique word
        Returns:
        - indexes: list -> list of index list for each unique word
        """
        indexes = []
        for word in self.unique_words:
            indexes_for_curr_word = []
            for i, row in self.reference_data.iterrows():
                phrase = row.get("reference", 0).split()
                if word in phrase:
                    indexes_for_curr_word.append(i)
            indexes.append(indexes_for_curr_word)
        return indexes

    def create_index_table(self):
        """
        Function to create index table
        Returns: 
        - index_table: pandas.core.frame.DataFrame -> dataframe with column Kata (unique words)
            and column Index (list of indexes, referring to Kata's value occurence(s) in reference data)
        """
        dct = {
            'Kata': self.unique_words,
            'Index': self.indexes
        }
        index_table = pd.DataFrame(dct)
        return index_table

    def get_unique_words(self):
        """
        Function to get list of unique words from reference data
        Returns:
        - unique_words: list -> list of unique words gathered from reference data
        """
        unique_words = []
        for index, row in self.reference_data.iterrows():
            phrase = row.get("reference", 0)
            words = phrase.split()
            for word in words:
                word = word.lower()
                if word not in unique_words:
                    unique_words.append(word)
        
        return unique_words

    def get_index_table(self):
        """
        Getter function to return the generated index table
        Returns:
        - self.index_table: pandas.core.frame.DataFrame
        """
        return self.index_table