import sys

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

import re
import numpy as np
import pandas as pd
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """ Loads data from the data base.
        Splits X(messages) and Y(categories) into distict lists
        
    Args:
        database_filepath: Filepath of the database
    Returns:
        two lists and one data frame
        X: a list of emergency messages 
        Y: a data frame includes category classifications of messages
        Column names: a list of category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('final_table',engine)
    X= df.message.values
    Y= df.drop(['id','message','original','genre'],1)
    Category_names=Y.columns
    return X, Y, Category_names


def tokenize(text):
    """ Tokenize given text:
        detects urls and replace them with usr_placholder
        Cleans stop words 
        Lemmatizes the text
    Args:
        Text: text string
    Returns:
        A list of clean tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    """Builds a machine learning pipeline.
    Creates a grid search parameters.
    Args: None
    Returns: 
        A machine learning model with grid search """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LinearSVC()))
    ])
  
    parameters = {
         
        'vect__max_df': (0.5, 0.75, 1.0)
        }

    cv=GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the machine learning algorithm
    Args:
        model: machine learning model
        X_test: test data of X values (messages)
        Y_test: test data of Y values(categories)
        category_names: a list of category names
     Returns:
        A data frame including precision, recall and f1 scores"""
    y_pred = model.predict(X_test)
    report= classification_report(Y_test,y_pred, target_names=category_names)
    temp=[]
    for item in report.split("\n"):
        temp.append(item.strip().split('     '))
    clean_list=[ele for ele in temp if ele != ['']]
    report_df=pd.DataFrame(clean_list[1:],columns=['group','precision','recall', 'f1-score','support'])
    return report_df


def save_model(model, model_filepath):
    """ Saves the trained model to a picle file
    Args:
        model: trained machine learnin model
        model_filepath: the filepath of the model
    Returns:
        None"""
    pickle.dump(model,open(model_filepath,'wb'))
 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()