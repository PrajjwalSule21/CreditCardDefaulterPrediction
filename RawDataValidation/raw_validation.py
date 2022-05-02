import warnings
warnings.filterwarnings('ignore')
import numpy as np
from Logger.logging import Log_class
from DataLoader.data_getter import GetData
from RawDataFeatureClassification.feature_classification import features
import os

class validation:

    def __init__(self):
        self.folder='F:/Ineuron_Internship/Project_CCDP/Log_Files/'
        self.filename='raw_validation.txt'
        self.df_object = GetData()
        self.features = features()

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

        self.log_obj = Log_class(self.folder,self.filename)

    def data_stdzero(self):

        """
                    Method: data_stdzero
                    Description: This method is used to check if standard deviation is Zero in the dataset
                    Parameters: None
                    Return: DataFrame after removing zero standard deviation columns

        """

        try:
            self.log_obj.log("INFO","Checking if the standard deviation is zero or not for numerical features ")
            dataframe = self.df_object.load_data()
            numerical_features = self.features.numerical_features()

            for feature in numerical_features:
                if dataframe[feature].std()== 0:
                    self.log_obj.log("INFO", "If there is a Zero std deviation columns are present then remove those columns")
                    dataframe.drop(columns=feature,axis=1,inplace=True)
            self.log_obj.log("INFO","Zero std deviation columns are removed")
            return dataframe

        except Exception as e:
            self.log_obj.log("ERROR","Occured Exception is :"+ "\t" + str(e))


    def data_whole_missing(self):

        """
                    Method: data_whole_missing
                    Description: This method is used to check if entire column has missing values
                    Parameters: None
                    Return: DataFrame after removing columns with entire missing values


        """

        try:
            self.log_obj.log("INFO","checking if there is column with whole missing values or not")
            data=self.data_stdzero()

            for i in data.columns:
                if data[i].isnull().sum()==len(data[i]):
                    self.log_obj.log("INFO", "If a column having whole missing value then drop it.")
                    data.drop(columns=i,axis=1,inplace=True)
                else:
                    pass

            self.log_obj.log("INFO","Whole missing value row is dropped")

            return data

        except Exception as e:
            self.log_obj.log("ERROR","Exception is:"+'\t'+str(e))

    def finding_null(self):
        """
                    Method: finding_null
                    Description: This method is used to check if their are any null values and replacing them with nan
                    Parameters: None
                    Return: DataFrame after checking null values

         """


        try:
            self.log_obj.log("INFO","Replacing null type values with NAN value")
            dataFrame=self.data_whole_missing()
            categorical=self.features.categorical_features()
            numerical=self.features.numerical_features()

            for feature in numerical:
                dataFrame[feature]=dataFrame[feature].replace(' ?',np.nan)
            self.log_obj.log("INFO", "Replace null type values with NAN in numerical column")


            for feature in categorical:
                dataFrame[feature]=dataFrame[feature].replace(' ?',np.nan)
            self.log_obj.log("INFO", "Replace null type values with NAN in categorical column")


            self.log_obj.log("INFO","Replaced all null type values with NAN values")
            return dataFrame

        except Exception as e:
            self.log_obj.log("ERROR","Exception is:"+'\t'+str(e))

