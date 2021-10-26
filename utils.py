# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from bert_sklearn import BertClassifier
from bert_sklearn import load_model

from nltk.corpus import stopwords
custom = stopwords.words("english")
custom.remove('not')


from sklearn.metrics import classification_report, f1_score

import pandas as pd
import os
import re
# import preprocessor
from bs4 import BeautifulSoup
import codecs
import numpy as np
from functools import partial

DATAPATH = '../data/data/'

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
url_pattern = re.compile('''((https?:\/\/)?(?:www\.|(?!www))[a-zA-Z0-9]([a-zA-Z0-9-]+[a-zA-Z0-9])?\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})''', 
                         flags=re.UNICODE)
mention_pattern = re.compile('([^a-zA-Z0-9]|^)@\S+', flags=re.UNICODE)
hashtag_pattern = re.compile('([^a-zA-Z0-9]|^)#\S+', flags=re.UNICODE)
rt_pattern = re.compile('([^a-zA-Z0-9]|^)(rt|ht|cc)([^a-zA-Z0-9]|$)', flags=re.UNICODE)







def detweet(text):
    return re.sub(url_pattern, '', 
               re.sub(rt_pattern, '', 
                      re.sub(mention_pattern, '',
                             re.sub(hashtag_pattern, '', 
                                 re.sub(emoji_pattern, '', 
                                    text)))))
def normalize(text):
    return re.sub(r"\s+", " ", #remove extra spaces
                  re.sub(r'[^a-zA-Z0-9]', ' ', #remove non alphanumeric, incl punctuation
                         text)).lower().strip() #lowercase and strip
def fix_encoding_and_unescape(text):
    return BeautifulSoup(text.decode('unicode-escape')).get_text()
def preprocess(text, fix_encoding=False):
    if (type(text)==str) or (type(text)==unicode):
        if fix_encoding:
            return normalize(detweet(fix_encoding_and_unescape(text)))
        else:
            return normalize(detweet(text))
    else:
        return text

def preprocess_light(tweets, fix_encoding=False):
    def _preprocess_light(text, fix_encoding=False):
        if (type(text)==str) or (type(text)==unicode):
            if fix_encoding:
                return normalize(re.sub(url_pattern, '',fix_encoding_and_unescape(text)))
            else:
                return normalize(re.sub(url_pattern, '',text))
        else:
            return text
    return map(partial(_preprocess_light, fix_encoding=fix_encoding), tweets)
    
def preprocess_jha2017(tweets, fix_encoding = False):
    """Replicates preprocessing as per Jha et al. 2017. 
    It removes usernames, punctuations, emoticons, hyperlinks/URLs and RT tag.
    It also lowercases, which seems a sensible thing to do.
    params: 
        tweets: list of unicode strings

    returns: list of unicode strings
    """
    return map(partial(preprocess, fix_encoding=fix_encoding), tweets)


class JhaPreprocessor(TransformerMixin):
    def __init__(self, fix_encoding=False):
        self.fix_encoding = fix_encoding
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return preprocess_jha2017(X, self.fix_encoding)

class IndiraPreprocessor(TransformerMixin):
        
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        import re
        import string
        from nltk.corpus import stopwords
        stops = set(stopwords.words('english'))
        stops.discard('not')

        def strip_links(text):
            link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
            links         = re.findall(link_regex, text)
            for link in links:
                text = text.replace(link[0], ', ')    
            return text

        def strip_all_entities(text):
            entity_prefixes = ['@','#']
            for separator in  string.punctuation:
                if separator not in entity_prefixes :
                    text = text.replace(separator,' ')
            words = []
            for word in text.split():
                word = word.strip()
                if word:
                    #print word
                    if word[0] not in entity_prefixes and word not in stops:
                        words.append(word.lower())
                    elif word in stops:
                        continue
                    else:
                        words.append('UNK')
            return ' '.join(words)
        return list(strip_all_entities(strip_links(t)) for t in X)


# different from the train fnction in train_models.py since it explicitly takes in counterfactual_proportion
def injected_train(training_data, counterfactual_data, models, construct = 'sentiment', counterfactual = False,
          preprocessing = None, model_name = 'logreg',
          labels = {'positive' : 1, 'negative' : 0}, one_sided = False,
          counterfactual_proportion = 0.5, 
          downsample_proportion = 1):
    """
    construct = ['sexism', 'sentiment' or 'hatespeech']
    counterfactual: [True, False]
    preprocessing = []
    model = ['logreg', 'bert', 'xgboost']
    """
    train_data = pd.read_csv(DATAPATH+training_data[construct]['original'], sep = '\t')
    
    if counterfactual:
        train_data = train_data.sample(int(len(train_data) * downsample_proportion))
        print("this is the amount of training data: ", len(train_data))
            
        if one_sided:
            # half the counterfactual_proportion since we only have one-sided counterfactuals
            counterfactual_proportion = counterfactual_proportion / 2
            
            # drop half of the non-sexist data
            non_sexist = train_data[train_data['sexism'] == 'non-sexist']
            #print(counterfactual_proportion, len(non_sexist), len(non_sexist)*(1-counterfactual_proportion))
            non_sexist_ = non_sexist.sample(n = int(len(non_sexist)*(1-counterfactual_proportion)))
            # replace with that much counterfactual
            extra = counterfactual_data.sample(n = int(len(non_sexist)*counterfactual_proportion))
            train_data = train_data[train_data['sexism'] == 'sexist'] # take full sexist data
            train_data = train_data.append(non_sexist_) # add original non-sexist data (1 - cf_prop) %
            train_data = train_data.append(extra) # add counterfactual data of cf_prop %
            
        else:
            train_data_ = train_data.sample(n = int(len(train_data)*(1-counterfactual_proportion)))
            extra = counterfactual_data.sample(n = int(len(train_data)*counterfactual_proportion))
            train_data = train_data_.append(extra)
                
        

    train_data[construct] = train_data[construct].map(labels)
#   print(train_data[construct].unique())
#     print(len(train_data))
#     train_data = train_data.dropna()
#     print("After dropping NaNs: ", len(train_data))
    
    
    if 'pipeline' in models[model_name]:
        grid_search = GridSearchCV(models[model_name]['pipeline'],
                               models[model_name]['params'],
                               n_jobs=-1,
                               verbose=1)
    else:
        grid_search = GridSearchCV(BertClassifier(validation_fraction=0),
                               models[model_name]['params'],
                               n_jobs=-1,
                               verbose=1)
    
    clf = grid_search.fit(train_data['text'], train_data[construct]).best_estimator_
     
    
    return clf


def get_results(cr, y_true, y_pred, 
                 method = 'logreg',
                 mode = False,
                 construct = 'sentiment',
                 labels = {1 : 'positive', 0 : 'negative'},
                 dataset = 'in-domain',
                 cf_type = 'all'):

    #print(cr)
    total = sum([cr[i]['support'] for i in labels.keys()])
    
    result = {}
    result['method'] = method
    result['dataset'] = dataset
    result['mode'] = mode
    result['construct'] = construct
    result['cf_type'] = cf_type
    
    for label in labels:

        result['Fraction of %s Class' %(label)] = float(cr[label]['support'])/total
        result['%s Class Precision' %(label)] = cr[label]['precision']
        result['%s Class Recall' %(label)] = cr[label]['recall']
        result['%s Class F1' %(label)] = cr[label]['f1-score']
        result['Fraction of Predicted %s' %(label)] = len([i for i in y_pred if i == label])\
                                            /len(y_pred)
    
    
    result['Macro Average Precision'] = cr['macro avg']['precision']
    result['Macro Average Recall'] = cr['macro avg']['recall']
    result['Weighted F1'] = f1_score(y_true, y_pred, average = 'weighted')
    result['Macro F1'] = cr['macro avg']['f1-score']
    
    
    return result

def generate_influential_examples(data, model, model_name = 'logreg', data_type = 'nCF', label = 'sentiment',
                                  mapping = {"positive"  :1, "negative" : 0}):

    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod import families

    import statsmodels.stats.tests.test_influence

    from sklearn.decomposition import TruncatedSVD


    grid_search = GridSearchCV(model['pipeline'],
                               model['params'],
                               n_jobs=-1,
                               verbose=1)
    data = data.sample(frac=1)
    
    if model_name == 'logreg':
        clf = grid_search.fit(data['text'], data.sentiment).best_estimator_
        vectorizer = clf.named_steps['tfidf']
        X = vectorizer.fit_transform(data.text)
        svd = TruncatedSVD(n_components=5, random_state=42)
        df = svd.fit_transform(X) 
    else:
        # load roberta embeddings
        pass
    df = pd.DataFrame(df)
    print(df)
    df['label'] = data.sentiment
    df['label'] = df['label'].map(mapping)
    df_ = df.loc[:, df.columns != 'label']
#     df_ = df_.loc[:, df_.columns != 'sample_type']
    df_.columns = [str(i) for i in df_.columns]
    
    res = GLM(df['label'], df_.astype(float),
          family=families.Binomial()).fit(attach_wls=True, atol=1e-10, )
    
    infl = res.get_influence(observed=False)
    summ_df = infl.summary_frame()
    
    summ_df['label'] = df['label']
    summ_df['sample_type'] = data['sample_type']
    summ_df['mode'] = [data_type] * len(summ_df)
    return summ_df.sort_values('cooks_d', ascending=False)    