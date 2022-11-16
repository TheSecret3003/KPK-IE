import pandas as pd
class PreprocessReferenceData(object):
    """
    Class PreprocessReferenceData to perform preprocessing of labeled BUMN and Kementerian data
    """
    def __init__(self):
        pass
    
    def create_reference_data_from_labeled(self, labeled_df):
        """
        Function to create a new reference data (yes labeled only, from column 'reference' - labeled data)
        Args:
        - labeled_df: pandas.core.frame.DataFrame -> DF of labeled data (manually labeled by the team)
        Returns:
        - res: pandas.core.frame.DataFrame -> reference data
        """
        res = labeled_df[labeled_df['status']=='yes'][['reference']]
        res = pd.DataFrame(res['reference'].unique())
        res = res.rename(columns={0: 'reference'})
        res = res.reset_index()
        res = res.drop(['index'], axis=1)
        # todo: clean data
        return res

    def create_reference_data_from_KPK(self, bumn_df, klpd_df):
        """
        Function to create a new reference data from KPK's list
        Args:
        - bumn_df: pandas.core.frame.DataFrame -> DF of BUMN lists
        - klpd_df: pandas.core.frame.DataFrame -> DF of KLPD lists
        Returns:
        - res: pandas.core.frame.DataFrame -> reference data
        """
        # The column of interest (nama instansi) in both DFs has been renamed to 'reference'
        # Assumption: all instansi are distinct to each other
        bumn_df = bumn_df[['reference']]
        klpd_df = klpd_df[['reference']]
        df = pd.concat([bumn_df, klpd_df])
        df = df.reset_index()
        df = df.drop(['index'], axis=1)
        #todo: clean data
        return df

    def combine_reference_data(self, labeled_df, bumn_df, klpd_df):
        """
        Function to concatenate the instansi references from manually labeled data and data from KPK
        """
        ref_data_from_labeled = self.create_reference_data_from_labeled(labeled_df)
        ref_data_from_KPK = self.create_reference_data_from_KPK(bumn_df, klpd_df)

        combined = pd.concat([ref_data_from_labeled, ref_data_from_KPK])
        combined = pd.DataFrame(combined['reference'].unique())
        combined = combined.rename(columns={0: 'reference'})
        combined = combined.reset_index()
        combined = combined.drop(['index'], axis=1)

        cleaned = self.clean_reference_data(combined)
        return cleaned
        
    def clean_reference_data(self, df):
        """
        Function to clean the passed df - removing punctuations
        Args:
        - df: pandas.core.frame.DataFrame -> reference data to be cleaned
        Returns:
        - df2: pandas.core.frame.DataFrame -> cleaned reference data
        """
        df2 = df.copy()
        df2['cleaned'] = df2['reference'].str.replace('/', ' ')
        df2['cleaned'] = df2['cleaned'].str.replace('[^\w\s]','',regex=True)
        df2 = pd.DataFrame(df2['cleaned'].unique())
        df2 = df2.rename(columns={0: 'reference'})
        df2 = df2.reset_index()
        df2 = df2.drop(['index'], axis=1)

        return df2