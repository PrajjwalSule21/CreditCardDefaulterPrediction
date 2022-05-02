from FeatureEngineering.data_scaling_transform import Data_scaling
from ModelBuilding.model_building import ModelBuilder



def training():
    """
    This is a training function for rawdata training and build the suitable model for the data.
    Parameter: None
    Return: None

    """
    try:

        scaling_obj=Data_scaling()
        scaling_obj.scaling()

        training_obj=ModelBuilder()

        training_obj.logictic_regression() # making the logistic model as a base line model

        model_name=training_obj.get_best_model() # findout the best model for the data


    except Exception as e:
        return "Error While training the ModelBuilding" +str(e)


if __name__ =='__main__':
    training()














