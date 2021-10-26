import warnings
warnings.filterwarnings('ignore')

import sys

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from bert_sklearn import BertClassifier
from bert_sklearn import load_model

from nltk.corpus import stopwords
custom = stopwords.words("english")
custom.remove('not')

from joblib import dump, load

from sklearn.metrics import classification_report
from utils import get_results, SocialPreprocessor

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

"""
define parameters
"""

DATAPATH = '../data/data/'
RESULTS = '../results/'
SAVEPATH = "../ml_models/"

constructs = [
               'sentiment',
               'sexism',
               'hatespeech'
               ]

test_types = ['in-domain',
              'out-domain',
             ]

adv_test_types = ['adv_inv',
               'adv_swap'
                 ]
runs = 1

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
                            'adv_inv' : 'sentiment/test/adv_inv.csv',
                            'adv_swap' : 'sentiment/test/adv_swap.csv',
                            
                           },
             'sexism' : {'in-domain' : 'sexism/test/original.csv',
                            'out-domain' : 'sexism/test/exist.csv',
                            'adv_inv' : 'sexism/test/adv_inv.csv',
                            'adv_swap' : 'sexism/test/adv_swap.csv',                            
                           },
             'hatespeech' : {'in-domain' : 'hatespeech/test/original.csv',
                            'out-domain' : 'hatespeech/test/hateval.csv',
                            'adv_inv' : 'hatespeech/test/adv_inv.csv',
                            'adv_swap' : 'hatespeech/test/adv_swap.csv',                            
                           },
            }

# to prevent confusion of which is the positive class

labels = {'sentiment' : {'positive': 1, 'negative' : 0},
          'sexism' : {'sexist' : 1, 'non-sexist' : 0},
          'hatespeech' : {'hate' : 1, 'not hate' : 0}
                          }


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
                   
                }

def train(construct = 'sentiment', counterfactual = False,
          preprocessing = None, model_name = 'logreg',
          labels = {'positive' : 1, 'negative' : 0}, one_sided = False,
          run = 0):
    """
    construct = ['sexism', 'sentiment' or 'hatespeech']
    counterfactual: [True, False]
    preprocessing = []
    model = ['logreg', 'bert', 'xgboost']
    """
    train_data = pd.read_csv(DATAPATH+training_data[construct]['original'], sep = '\t')
    
    if counterfactual:
        if one_sided:
            # drop half of the non-sexist data
            non_sexist = train_data[train_data['sexism'] == 'non-sexist']
            non_sexist = non_sexist.sample(n = len(non_sexist)//2)
            # replace with that much counterfactual
            extra = pd.read_csv(DATAPATH+training_data[construct]['counterfactual'],
                                sep = '\t').sample(n = len(non_sexist))
            train_data = train_data[train_data['sexism'] == 'sexist'] # take full sexist data
            train_data = train_data.append(non_sexist) # add half original non-sexist data
            train_data = train_data.append(extra) # add half counterfactual data
            
        else:
            train_data = train_data.sample(n = len(train_data)//2)
            extra = pd.read_csv(DATAPATH+training_data[construct]['counterfactual'],
                                sep = '\t').sample(n = len(train_data))
            train_data = train_data.append(extra)
                
        
        
    train_data[construct] = train_data[construct].map(labels)
    print(counterfactual, len(train_data), train_data[construct].unique())
    
    if 'pipeline' in models[model_name]:
        grid_search = GridSearchCV(models[model_name]['pipeline'],
                               models[model_name]['params'],
                               n_jobs=-1,
                               verbose=1)
        clf = grid_search.fit(train_data['text'], train_data[construct]).best_estimator_
    
    else:
        models[model_name]['output_dir'] = SAVEPATH + "/%s_%s_%d" %(construct, counterfactual, (1)
          )
        
        model_args = ClassificationArgs(num_train_epochs=models[model_name]['num_train_epochs'],
                                         output_dir = models[model_name]['output_dir'])
        clf = ClassificationModel(
                   "bert", "bert-base", args = model_args, use_cuda = False
                    )

        train_data['labels'] = train_data[construct]
        clf.train_model(train_data)
    
    
     
    
    return clf

# only for checking, should have seperate notebook for tests alone

def test(model, data, construct = 'sentiment', test = 'in-domain',
         labels = {'positive' : 1, 'negative' : 0}, preprocessing = None, 
         threshold = 0.5, predicted = 'predicted', text_column = 'text'):
    
    data = data[data[construct].isin(labels.keys())]
    data[construct] = data[construct].map(labels)
    data['predicted'] = model.predict(data[text_column])
#    thresholded = (model.predict_proba(data[text_column]) >= threshold).astype(int).argmax(axis = 1)
    
#    data['threshold predicted'] = [labels[i] for i in thresholded]
    
#     print(data['predicted'][0:5])
#     print(data['threshold predicted'][0:5])
    
    return list(data[construct]), list(data[predicted]),\
            classification_report(data[construct], data[predicted], output_dict = True)


if __name__ == "__main__":
    
    trained_models = {}

    for construct in constructs:
        trained_models[construct] = {}
        for run in range(0, 1):
            for model_name in models.keys():
                for mode in [False, True]: #change and add counterfactuals as well
                    if model_name not in trained_models[construct]:
                        trained_models[construct][model_name] = {True: [], False: []}
                    if construct == 'sexism':
                        one_sided = True
                    else:
                        one_sided = False
                    trained_models[construct][model_name][mode].append(train(construct, 
                                                              model_name = model_name,
                                                              counterfactual = mode,
                                                              labels = labels[construct],
                                                              run = run,
                                                              one_sided = one_sided))
                    if model_name != 'bert': # bert models are automatically saved
                        dump(trained_models[construct][model_name][mode][run],
                         '../ml_models/%s_%s_%s_%d.joblib' %(construct, model_name, mode, run)) 
        
        
    print(trained_models)

    # test all models 

    
    sys.stdout = open(RESULTS + "/train_models_results.txt","a")
    
    all_results = []

    for construct in constructs:
        for run in range(runs):
            for model_name in models.keys():
                for mode in [False, True]:
                    for test_type in test_types:
                        data = pd.read_csv(DATAPATH+test_data[construct][test_type], sep = '\t')
                
                        #print()
                        #print(construct, model_name, mode, test_type)
                        true, pred, cr = test(trained_models[construct][model_name][mode][run],
                                          data, construct, test = test_type,
                                          labels = labels[construct])
                        all_results.append(get_results(cr, true, pred,
                                              method = model_name,
                                              mode = mode,
                                              construct = construct,     
                                              labels = {str(v): k for k, v in labels[construct].items()}, #invert label mapping
                                              dataset = test_type))

    
    

    results = {}

    result_df = pd.DataFrame(all_results)
    result_df['mode'] = result_df['mode'].map({True: "Counterfactual", False: "Non-Counterfactual"})
    for construct in constructs:
        result_df_ = result_df[result_df['construct'] == construct]
        results[construct] = result_df_.groupby(['construct','method', 'dataset', 'mode'])\
        [['Macro Average Precision', 'Macro Average Recall', 'Macro F1']].mean().unstack()                    


    
    
        results[construct].to_csv(RESULTS + "/%s_in_out_domain_results.csv" %(construct), sep = "\t")
    
        print("\n\n\n\nIn-domain and Out-of-domain results for %s\n\n" %(construct))
        print(results[construct].round(3).to_latex())
        print("\n\n\n\n")


    # test for adversarial examples but also test on their original counterparts 
    # to prevent data size discrepencies

    all_results = []

    for construct in constructs:
        for run in range(runs):
            for model_name in models.keys():
                for mode in [False, True]:
                    for test_type in adv_test_types:
                        data = pd.read_csv(DATAPATH+test_data[construct][test_type], sep = '\t')
                
                        #print()
                        #print(construct, model_name, mode, test_type)
                    
                        print()
                        print(construct, model_name, mode, test_type)
                    
                        # first the original examples
                        true, pred, cr = test(trained_models[construct][model_name][mode][run],
                                          data, construct, test = test_type + " original",
                                          labels = labels[construct], text_column = 'original')
                        all_results.append(get_results(cr, true, pred,
                                              method = model_name,
                                              mode = mode,
                                              construct = construct,     
                                              labels = {str(v): k for k, v in labels[construct].items()},
                                              dataset = test_type + " original"))
                    
                        # second the adversarial examples
                        true, pred, cr = test(trained_models[construct][model_name][mode][run],
                                          data, construct, test = test_type,
                                          labels = labels[construct])
                        all_results.append(get_results(cr, true, pred,
                                              method = model_name,
                                              mode = mode,
                                              construct = construct,     
                                              labels = {str(v): k for k, v in labels[construct].items()},
                                              dataset = test_type))

    results = {}
    result_df = pd.DataFrame(all_results)
    result_df['mode'] = result_df['mode'].map({True: "Counterfactual", False: "Non-Counterfactual"})
    for construct in constructs:
        result_df_ = result_df[result_df['construct'] == construct]
        results[construct] = result_df_.groupby(['construct','method', 'dataset', 'mode'])\
        [['Macro Average Precision', 'Macro Average Recall', 'Macro F1']].mean().unstack()                    
    
        results[construct].to_csv(RESULTS + "/%s_adversarial_results.csv" %(construct), sep = "\t")
    
        print("\n\n\n\nPerformance on Adversarial Examples for %s\n\n\n" %(construct))
        print(results[construct].round(3).to_latex())
        print("\n\n\n\n\n")
    

    sys.stdout.close()
    