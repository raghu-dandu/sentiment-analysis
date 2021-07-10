# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:01:44 2020

@author: Raghu
"""

#Library import and file reading

#import necessary libraries
import numpy as np
import pandas as pd
import nltk
import re

def process_twitter_text(data):
    data=[re.sub(r'&[A-Za-z0-9]+','',text) for text in data]
    data=[re.sub(r'@[A-Za-z0-9]+','',text) for text in data]
    data=[re.sub('https?://[A-Za-z0-9./]+','',text) for text in data]
    data=[re.sub('[^A-Za-z0-9 ]+', '', text) for text in data]
    data=[re.sub('https?://[A-Za-z0-9./]+','',text) for text in data]
    data=[re.sub(' +', ' ', text) for text in data]
#    data=[text.lower() for text in data]
    return data

def calculate_accuracy(folder_name):
    #Calculate accuracy
    sample_filename = folder_name+"/"+ "Tweets_sentiment_inp.csv"
    test_filename = folder_name+"/"+ "Tweets_sentiment_test_input.csv"
   
    testing_df = pd.read_csv(test_filename)
    sample_df=pd.read_csv(sample_filename)
    from sklearn.metrics import accuracy_score
    print("Accuracy : ",accuracy_score(testing_df['Customer sentiment / tone'][0:50],sample_df['Customer sentiment / tone'][0:50]))
   
    from sklearn.metrics import confusion_matrix
    cf_matrix = confusion_matrix(testing_df['Customer sentiment / tone'][0:50],sample_df['Customer sentiment / tone'][0:50])
    print("Confusion Matrix: ", cf_matrix)
    import seaborn as sns
    sns.heatmap(cf_matrix, annot=True)
    from sklearn.metrics import classification_report
    print("Classification Report: ", classification_report(testing_df['Customer sentiment / tone'][0:50],sample_df['Customer sentiment / tone'][0:50]))
    
if __name__=="__main__":
    #Read file location from console
    folder_name = input("Enter folder name(Should contain the inputfile i.e ..\POC deliverables\Datasets):")
    folder_name.strip()
    folder_name.strip("/")
    full_filename = folder_name+"/"+ "Telco_twitter_data.csv"
    #Read Customer tweets  - input file
    twitter_df = pd.read_csv(full_filename)
    ##Pre-processing text data
    
    #Delete rows with empty text
    twitter_df.dropna(subset = ["Tweet"], inplace=True)
    
    #Select on English language
    twitter_df_eng=twitter_df.loc[twitter_df['Language'] == 'English'] 
    
    tweets_df=twitter_df_eng[['Tweet']]
    
    #Get Sentiment polarizations
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    
    tweets_df['Polarity_scores'] = tweets_df['tweet_processed'].apply(lambda Text: sid.polarity_scores(Text))
    tweets_df['Sentiment Score']  = tweets_df['Polarity_scores'].apply(lambda score_dict: score_dict['compound'])
    tweets_df['Customer sentiment / tone'] = tweets_df['Sentiment Score'].apply(lambda s: 'positive' if s >=0.05 else('negative' if s <= -0.05 else 'neutral'))    
    #drop Polarity_scores and compound_score columns
    tweets_df.drop(['Tweet','Polarity_scores','tweet_processed'], axis=1, inplace=True)
    tweets_sentiment_df = pd.concat([twitter_df_eng, tweets_df], axis=1)
   
    out_file= folder_name+"/output/"+ "Tweets_sentiment_output.csv"
    tweets_sentiment_df.to_csv(out_file,index=False)
    print("Output written to ",out_file)
    calculate_accuracy(folder_name)
    
 