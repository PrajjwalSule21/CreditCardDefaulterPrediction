import warnings
warnings.filterwarnings('ignore')
from Logger.logging import Log_class
from sklearn.ensemble import ExtraTreesClassifier
from RawDataPreprocessing.preprocessing import Preprocessing
import os

class FeatureImportance:
    def __init__(self):
        self.folder = 'F:/Ineuron_Internship/Project_CCDP/Log_Files/'
        self.filename = 'FeatureSelection.txt'
        self.preprocessing = Preprocessing()
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_obj = Log_class(self.folder, self.filename)


    def feature_importance(self):
        """
                            Method: feature_importance
                            Description: This method is used to select the importace features for label, so that model will give accurate results
                            Parameters: None
                            Return: DataFrame after selecting the features


                """

        try:
            self.log_obj.log('INFO', "Finding features importance")
            dataframe = self.preprocessing.remove_duplicates()
            self.log_obj.log('INFO', "Make the object of ExtraTreesClassifier() for getting the importance features")
            model = ExtraTreesClassifier()

            self.log_obj.log('INFO', "Making the TOTAL_BILL_PAY from all 6 month BILL_AMT features")
            dataframe['TOTAL_BILL_AMT'] = ''
            dataframe['TOTAL_BILL_AMT'] = dataframe['BILL_AMT1']+ dataframe['BILL_AMT2'] + dataframe['BILL_AMT3'] + dataframe['BILL_AMT4'] + dataframe['BILL_AMT5'] + dataframe['BILL_AMT6']

            self.log_obj.log('INFO', "Making the TOTAL_PAY_AMT from all 6 month PAY_AMT features")
            dataframe['TOTAL_PAY_AMT'] = ''
            dataframe['TOTAL_PAY_AMT'] = dataframe['PAY_AMT1'] + dataframe['PAY_AMT2'] + dataframe['PAY_AMT3'] + dataframe['PAY_AMT4'] + dataframe['PAY_AMT5'] + dataframe['PAY_AMT6']

            self.log_obj.log('INFO', "Droping the individual BILL_AMT and PAY_AMT features from dataframe")
            dataframe.drop(columns=['BILL_AMT1', 'BILL_AMT2',
                               'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                               'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'], inplace=True)



            self.log_obj.log('INFO', 'Selecting the X and y for ExtraTreesClassifier() model')
            X = dataframe.drop(columns=['ID','DEFAULT_PAYMENT_NEXT_MONTH'])  # independent columns
            y = dataframe['DEFAULT_PAYMENT_NEXT_MONTH']  # target column i.e DEFAULT_PAYMENT_NEXT_MONTH

            self.log_obj.log('INFO', 'Fitting the model on X and y')
            model.fit(X,y)

            importantfeatures = model.feature_importances_
            self.log_obj.log('INFO', f'The important feature are:{importantfeatures}')

            self.log_obj.log('INFO', 'Selecting the Important featues and make a dataframe of those featues')
            dataframe = dataframe[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'TOTAL_BILL_AMT', 'TOTAL_PAY_AMT', 'DEFAULT_PAYMENT_NEXT_MONTH']]

            return dataframe

        except Exception as e:
            self.log_obj.log('ERROR', 'Error occurs while selecting the feratures' + str(e))


