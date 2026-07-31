"""
- this will take the data from the defined source. 
- for the scope of this project we will pull the data hosted on github. In real life the data 
   might be present in any DBMS server, or in S3 bucker or any Azure Blobs. There we will use 
   authentication credentials in the data_ingestion module in order to pull the data. 

"""



import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml 

##########LOGGING_MODULE##########

#Ensure that the logs directory exists 
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#lgging configuration

#here we are seting up the logger object and naming it data_ingestion
logger = logging.getLogger('data_ingestion')
#setting the log evel to debug. Sara info de dena
logger.setLevel('DEBUG')

#here we are setting the handler -> console handler 
#console handler prints the logs in the terminalllok
console_handler = logging.StreamHandler()
#the console handler that we have made use bhi logging level debug hi diya
console_handler.setLevel('DEBUG')

#here we are setting the Filehandler. Since FileHandler writes the logs in the file 
#so we will first define the location of our log file
log_file_path = os.path.join(log_dir,'data_ingestion.log')
#here we define the Filehandler and we have also sssigned the log level DEBUG to it
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter('DEBUG')

#here we are setting the formatter
#time - loggger_name  - log_level - message 
#this is the format we will use to log our results
#and we will use the same formatter for both our Filehandler as well as ConsoleHandlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

#sincve we know that the logger object is formed by both handler and formatter objects 
#so here we are defining the logger obj by providing both handler-> console and filehandler 
#as well as the formatter
logger.addHandler(console_handler)
logger.addHandler(file_handler)

##########LOGGING_MODULE##########

def load_params(params_path: str)->dict:
    """load parameters from a YAML file
    """
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s',params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Unexpected error: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s',e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a csv file
    """
    try:
        df = pd.read_csv(data_url)
        #after the data is loaded then accordiingly whether the operation is successful or not 
        #logger object that we have created above will assign the logs by printing this mesg
        #it is using debug level priority
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('failed to parse the csv file: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured while loading the data: %s',e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data
    """
    try:
        df.drop(columns = ['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1':'target','v2':'text'}, inplace = True)
        logger.debug('Data processing completed')
        return df
    except KeyError as e:
        logger.error('Missing columns in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during the preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """save the train and test datasets
    """
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok = True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index = False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index = False)
        logger.debug('Train and test data are saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occured while saving the data: %s',e)
        raise

def main():
    try:
        params = load_params(params_path = 'params.yaml')
        test_size = params['data_ingestion']['test_size']
        #test_size = 0.2
        data_path = 'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
        df = load_data(data_url = data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size = test_size, random_state = 2)
        #data/raw_data/train,test
        save_data(train_data,test_data, data_path = './data')
    except Exception as e:
        logger.error('failed to complete the data ingestion process: %s', e)
        print(f'Error: {e}')



if __name__ == '__main__':
    main()
