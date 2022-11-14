class PreprocessReferenceData(object):
    """
    Class PreprocessReferenceData to perform preprocessing of labeled BUMN and Kementerian data
    """
    def __init__(self, df):
        self.ori_df = df
        self.reference_data = self.create_reference_data()
    
    def create_reference_data(self):
        """
        Function to create a new reference data (yes labeled only, from column 'reference')
        Returns:
        - res: pandas.core.frame.DataFrame -> reference data
        """
        res = self.ori_df[self.ori_df['status']=='yes'][['reference']]
        res = res.reset_index()
        res = res.drop(['index'], axis=1)
        res = res.drop_duplicates()
        return res

    def get_preprocessed_df(self):
        """
        Getter function to return the generated reference data
        Returns:
        - self.reference_data: pandas.core.frame.DataFrame
        """
        return self.reference_data