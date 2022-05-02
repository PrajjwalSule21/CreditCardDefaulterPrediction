import warnings

warnings.filterwarnings('ignore')
from Logger.logging import Log_class
from RawDataValidation.raw_validation import validation
from RawDataFeatureClassification.feature_classification import features
import os


class Preprocessing:
    def __init__(self):
        self.folder = 'F:/Ineuron_Internship/Project_CCDP/Log_Files/'
        self.filename = 'preprocessing.txt'
        self.validation = validation()
        self.features = features()

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_obj = Log_class(self.folder, self.filename)

    def fill_missing_values(self):

        """
                    Method: fill_missing_values
                    Description: This method is used to filling missing values present in the dataset
                    Parameters: None
                    Return: DataFrame after filling the missing values


        """

        try:
            self.log_obj.log('INFO', "Finding features with missing values")
            dataframe = self.validation.finding_null()
            numerical = self.features.numerical_features()
            categorical = self.features.categorical_features()

            misssing_num_features = [i for i in numerical if dataframe[i].isnull().sum() > 0]
            misssing_cat_features = [i for i in categorical if dataframe[i].isnull().sum() > 0]

            self.log_obj.log("INFO", f"The missing numerical features are: {misssing_num_features}")
            self.log_obj.log("INFO", f"The missing categorical features are: {misssing_cat_features}")

            for feature1 in misssing_cat_features:
                dataframe[feature1].fillna(dataframe[feature1].mode()[0], inplace=True)

            for feature2 in misssing_num_features:
                dataframe[feature2].fillna(dataframe[feature2].mean(), inplace=True)

            self.log_obj.log("INFO", "Missing values imputaion is done")

            return dataframe

        except Exception as e:
            self.log_obj.log("ERROR", "The Exception is:" + '\t' + str(e))

    def remove_duplicates(self):

        """
                    Method: remove_duplicates
                    Description: This method is used to remove duplicates  values if present in the dataset
                    Parameters: None
                    Return: DataFrame after removing the duplicate values


        """

        try:
            self.log_obj.log("INFO", "Removing Duplicates from Dataframe")
            data = self.fill_missing_values()

            data.drop_duplicates(keep='first', inplace=True)

            self.log_obj.log("INFO", "Duplicate values have been dropped")

            return data


        except Exception as e:
            self.log_obj.log("ERROR", "Exception is:" + '\t' + str(e))



