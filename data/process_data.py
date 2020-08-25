import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """loads two data files and merges them into one 
    Args
        messages_filepath: first data file's path. This includes emergency messages
        categories_filepath: second data file's path. This includes categories assigned to each message.
    
    Returns
        One merged data frame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on ='id')
    return df

def clean_data(df):
    """Cleans the given data frame:
        Splits categories into distinct columns. 
        Converts category values to 0 and 1.
        Drops the colums have all zero values.
        Assigns value of 1 to the ones whose max values exceeds 1
        Drops duplicate rows.
        Drops row whose entire columns is Na.
        
    Args:
        df: data frame created by load_data function
    Returns:
        Clean data frame"""
    
    categories = df['categories'].str.split(pat=';',expand=True)
    row = categories.iloc[0,]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[len(x)-1:])  # set each value to be the last character of the string
        categories[column] = categories[column].apply(int) # convert column from string to numeric
    
    for column in categories: #drops the colums have all zero values and fixes max values exceeds 1
        if categories[column].max()==0:
           categories=categories.drop(column,1)
        elif categories[column].max()>1:
            categories.loc[categories[column]>1,column]=1
    
    df = df.drop('categories',1)
    df = pd.concat([df, categories], axis=1).reindex(df.index)
    df = df.drop_duplicates()
    df=df.dropna(how='all')
    return df

def save_data(df, database_filename):
    """Saves the data frame to a defined sql data base
    Args:
        df: data frame to be saved to the database
        database_filename: database file name df will be stored
    Returns:
        None"""
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('final_table', engine, index=False, if_exists='replace')
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()