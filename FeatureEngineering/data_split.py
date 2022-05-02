from Logger.logging import Log_class
from FeatureSelection.feature_selection import FeatureImportance
import os
import warnings
warnings.filterwarnings('ignore')

class split:

    def __init__(self):
        self.folder='F:/Ineuron_Internship/Project_CCDP/Log_Files/'
        self.filename='splitting_data.txt'
        self.df_obj = FeatureImportance()


        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_obj=Log_class(self.folder,self.filename)


    def splitting(self):

        """
                    Method: splitting
                    Description: This method is used to split the data into dependent and Independent variables
                    Parameters: None
                    Return: Independent and dependent variables in the form of X and y, where X represents all the independent variables and y represents the dependent varibale.

        """

        try:
            self.log_obj.log("INFO","Splitting the data into X and y")
            data = self.df_obj.feature_importance()
            x = data.drop('DEFAULT_PAYMENT_NEXT_MONTH',axis=1)
            self.log_obj.log("INFO", "Variable X contains all the Features")
            y = data['DEFAULT_PAYMENT_NEXT_MONTH']
            self.log_obj.log("INFO", "Variable y contains the Label")


            self.log_obj.log("INFO","splitting of Features and Label has been done")

            return x,y

        except Exception as e:
            self.log_obj.log("INFO","Exception occured is:"+'\t'+str(e))



