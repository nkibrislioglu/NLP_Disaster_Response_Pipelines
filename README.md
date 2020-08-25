# Disaster Response Pipeline Project

## Table of Contents
* Libraries
* Project Motivation
* File Descriptions
* Instructions
* Licensing, Authors, and Acknowledgements

## Libraries
The libraries used in this repository are:
* pandas
* numpy
* matplotlib
* nltk
* sklearn

## Project Motivation
The purpose of this project is to classify messages that send in an emergency to help first responders. Data set consists of emergency messages and their classifications. One message can be classified on more than one category.
The project includes and an web app which enables us to categorize new messages with existing algorithm


## File Descriptions
There are three folders in this repository:
* app folder: includes the codes for creating a web app that shows classification of new messages
* data folder: includes data files, database and data preprocessing codes
* models file: includes model training codes and model pickle file


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Authors, and Acknowledgements
This repository is part of Udacity Data Science Nano degree program projects



