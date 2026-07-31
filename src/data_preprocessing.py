import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Define logging mechanism
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setup logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler
log_file_path = os.path.join(log_dir, 'data_preprocessing_log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def transform_text(text):
    """Transforms the input text by converting it to lowercase, tokenizing, removing 
    stopwords and punctuation, and applying stemming.
    """
    ps = PorterStemmer()
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words and word not in string.punctuation]
    
    # Stem the words
    text = [ps.stem(word) for word in text]
    
    # Join the tokens back into a single string
    return " ".join(text)

def preprocess_df(df, text_column='text', target_column='target'):
    """Preprocesses the dataframe by encoding the target column, removing duplicates, 
    and transforming the text column.
    """
    try:
        logger.debug('Starting preprocessing for the DataFrame')
        
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Remove duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')

        # Apply text transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        
        return df
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

def main(text_column='text', target_column='target'):
    """Main function to load the data, preprocess it, and save the processed data."""
    try:
        # Load data from raw directory
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Save processed data
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()