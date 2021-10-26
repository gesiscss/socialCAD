import pandas as pd
import numpy as np
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from bert_sklearn import BertClassifier
from bert_sklearn import load_model

from nltk.corpus import stopwords
custom = stopwords.words("english")
custom.remove('not')

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams.update({'font.size': 18})

import pickle

from utils import SocialPreprocessor, injected_train
from joblib import dump, load

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

DATAPATH = '../data/data/'
GENDERED_VOCAB_REL_PATH='gendered_vocabulary/gender_words.txt'
constructs = ['sentiment', 'sexism', 'hatespeech']
SAVEPATH = "../ml_models/typology/"


models = {
          'logreg' : {'pipeline' : Pipeline([
                        ('preprocess', SocialPreprocessor()),
                        ('tfidf', TfidfVectorizer()),
                        ('clf', LogisticRegression()),
                    ]),
                     'params' : {
                        'tfidf__stop_words': ['english', None, custom],
                        'tfidf__norm': ('l1', 'l2'),
                        'clf__max_iter': (20,),
                        'clf__C': (0.01, 0.1, 1),
                        'clf__penalty': ('l2', 'l1'),
                    }
                },
          'bert' : {'num_train_epochs' : [4, 5],
                    'output_dir' : SAVEPATH,
                    'learning_rate' : [2e-5, 3e-5, 5e-5]
                    }


training_data = {'sentiment': {'original' : 'sentiment/train/original.csv',\
                               'counterfactual' : 'sentiment/train/counterfactual.csv'
                              },
            'sexism': {'original' : 'sexism/train/original.csv',\
                               'counterfactual' : 'sexism/train/counterfactual.csv'
                              },
            'hatespeech' : {'original' : 'hatespeech/train/original.csv',\
                               'counterfactual' : 'hatespeech/train/counterfactual.csv'
                              },
                }

test_data = {'sentiment' : {'in-domain' : 'sentiment/test/original.csv',
                            'out-domain' : 'sentiment/test/kaggle.csv',
                            #'adv_inv' : 'sentiment/test/adv_inv.csv',
                            #'adv_swap' : 'sentiment/test/adv_swap.csv',
                            
                           },
             'sexism' : {'in-domain' : 'sexism/test/original.csv',
                            'out-domain' : 'sexism/test/exist.csv',
                            #'adv_inv' : 'sexism/test/adv_inv.csv',
                            #'adv_swap' : 'sexism/test/adv_swap.csv',                            
                           },
             'hatespeech' : {'in-domain' : 'hatespeech/test/original.csv',
                            'out-domain' : 'hatespeech/test/hateval.csv',
                            #'adv_inv' : 'hatespeech/test/adv_inv.csv',
                            #'adv_swap' : 'hatespeech/test/adv_swap.csv',                            
                           },
            }

labels = {'sentiment' : {'positive': 1, 'negative' : 0},
          'sexism' : {'sexist' : 1, 'non-sexist' : 0},
          'hatespeech' : {'hate' : 1, 'not hate' : 0}
                          }

data = {}
for construct in constructs:
    data[construct] = pd.read_csv("../data/data/%s/train/paired.csv" %(construct), sep = "\t")


types = ['negation_additions',
       'negation_deletions', 'affect word_additions', 'affect word_deletions',
       'gender word_additions', 'gender word_deletions',
       #'identity word_additions', 'identity word_deletions',
        'hedges_additions', 'hedges_deletions',
        'hate words_additions', 'hate words_deletions',] 

unidirectional_types = ['all', #all---> random selection of counterfactuals
       'negation',
       'affect word',
       'gender word',
       #'identity word',
       'hedges',
        'hate words'] 

strategies = [
				True,
				False,
				'mixed'
				 ]

# for this analysis, as we are capped by a maximum of 20% for some counterfactual types, we will go up 20% only
proportions = range(4, 1, -1)
multiplier = 0.05
runs = 1


dist_dict = {}
dist_list = []

for construct in constructs:
    df = data[construct]
    dist_dict = {}
    dist_dict['construct'] = construct
    dist_dict['total'] = len(df)
    total = len(df)
    print("total examples of " + construct + ": " + str(total))
    for diff_type in unidirectional_types:
        if diff_type != 'all':
            data_ = df[(df['%s_additions' %(diff_type)] == True) | (df['%s_deletions' %(diff_type)] == True)]
            dist_dict[diff_type] = len(data_)/total
            print(diff_type + ": " + str(len(data_)/total))
    print()
    dist_list.append(dist_dict)

# change the diff distribution to construct-driven and construct-agnostic
construct_driven_diffs = {'sentiment' : 'affect word',
                         'sexism' : 'gender word',
                         'hatespeech' : 'hate words'}

for construct in constructs:
    data[construct]['construct-driven'] = [True if row['%s_additions' %(construct_driven_diffs[construct])]\
                                           or row['%s_deletions' %(construct_driven_diffs[construct])]\
                                           else False for n, row in data[construct].iterrows()]
    

trained_models = {}  # construct ---> model_name ---> mode ---> stategy ---> prop X runs

for construct in ['sexism']:#, 'hatespeech']:
    trained_models[construct] = {}
    
    for model_name in models.keys():
        trained_models[construct][model_name] = {}
        
        for mode in [True]: #change and add counterfactuals as well
            trained_models[construct][model_name][mode] = {}
                
            # we train a model only on construct-driven counterfactuals, and one on construct-agnostic
            for prop in proportions: # prop == 0 means the model is essentially non-counterfactual
                actual_prop = prop * multiplier 
                
                trained_models[construct][model_name][mode][actual_prop] = {} 
                for strategy in strategies:
                    data_ = data[construct]
                    if strategy == 'mixed':
                        min_class_size = data_.groupby('construct-driven').size().min()
                        data_ = data_.groupby('construct-driven', group_keys=False)\
                        .apply(lambda x: x.sample(n=min_class_size, replace=False)).reset_index(0, drop=True)
                    else:
                        data_ = data_[data_['construct-driven'] == strategy]
                
                    counterfactual_data = data_[['counterfactual_id', 'counterfactual_text', 'counterfactual_label']]
                    counterfactual_data.columns = ['_id', 'text', construct]
                    
                    print(len(counterfactual_data))
                    
                

                    trained_models[construct][model_name][mode][actual_prop][strategy] = []
                    
                    for run in range(runs):
                        if construct == 'sexism':
                            one_sided = True
                        else:
                            one_sided = False
                        if True:
                            trained_models[construct][model_name][mode][actual_prop][strategy].append(injected_train(training_data, counterfactual_data = counterfactual_data, 
                                                     models = models, construct = construct, model_name = model_name,
                                                     counterfactual = mode, labels = labels[construct],
                                                     one_sided = one_sided, counterfactual_proportion = actual_prop,
                                                     # downsample_proportion = 0.5
                                                    )) 
                            

                            
                            if model_name != 'bert': # bert models are automatically saved
                                dump(trained_models[construct][model_name][mode][run],
                                '../ml_models/%s_%s_%s_%d.joblib' %(construct, model_name, mode, run)) 
#                         except Exception as e:
#                             # SHOULD NOT HAPPEN
#                             print(e)
#                             trained_models[construct][model_name][mode][strategy][actual_prop].append('not enough counterfactual data')
#                             pass
