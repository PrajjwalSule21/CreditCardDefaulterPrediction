from DataBase.database import DBoperation

"""
This is a separate script for extracting the raw data from database, 
and inserting the features table into database for further use. 

This script runs separately because the concern of latency.
"""

path = 'F:/Ineuron_Internship/Project_CCDP/DataSet/'

database = DBoperation()
# This is the object for DBoperation

database.dataextraction(path)
# This will extract data from Database and save a CSV file at desired location

# database.datainsertion()
# This will insert the useful features of data after preprocesssing.