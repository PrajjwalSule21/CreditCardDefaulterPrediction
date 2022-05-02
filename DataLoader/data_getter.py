import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from Logger.logging import Log_class
import os


class GetData:

    def __init__(self):
        self.training_file = 'F:/Ineuron_Internship/Project_CCDP/DataSet/CustomerCreditCard.csv'
        self.folder = 'F:/Ineuron_Internship/Project_CCDP/Log_Files/'
        self.file_name = "loading_raw_data.txt"

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_obj = Log_class(self.folder, self.file_name)

    def load_data(self):

        """
                    Method: load_data
                    Description: This method is used to load the dataset
                    Parameters: None
                    Return: Gave the DataFrame of a data.

        """

        try:
            self.log_obj.log("INFO", "loading the dataset into pandas dataframe")
            self.dataframe = pd.read_csv(self.training_file)
            self.log_obj.log("INFO", 'Raw data got loaded in dataframe')

            return self.dataframe

        except Exception as e:
            self.log_obj.log("ERROR", "Failed while loading dataset.Error is:" + str(e))


