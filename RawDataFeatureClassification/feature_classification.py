import warnings
warnings.filterwarnings('ignore')
from Logger.logging import Log_class
import os
from DataLoader.data_getter import GetData

class features:

    def __init__(self):
        self.folder = 'F:/Ineuron_Internship/Project_CCDP/Log_Files/'
        self.file_name = "feature_classification_logs.txt"
        self.df_object = GetData()

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_obj=Log_class(self.folder,self.file_name)

    def numerical_features(self):

        """
                       Method:numerical_features
                       Description:This method is used to get numerical features of the given dataset
                       Parameters:None
                       Return:Numerical variables of the dataset


        """

        try:
            dataframe = self.df_object.load_data()
            numerical = [i for i in dataframe.columns if
                         dataframe[i].dtypes == 'int64' or dataframe[i].dtypes == 'float64']
            self.log_obj.log("INFO", f"Numerical features in dataset are: {numerical}")
            return numerical

        except Exception as e:
            self.log_obj.log("ERROR", "Exception caused.Error is:" + str(e))

    def categorical_features(self):

        """
                        Method: categorical_features
                        Description: This method is used to get categorical features of the given dataset
                        Parameters: None
                        Return: categorical variables of the dataset


        """

        try:
            dataframe = self.df_object.load_data()
            categorical = [i for i in dataframe.columns if dataframe[i].dtypes == 'O']
            self.log_obj.log("INFO", f"Categorical features are:{categorical}")
            return categorical

        except Exception as e:
            self.log_obj.log("ERROR", "Exception caused.Error is:" + str(e))



