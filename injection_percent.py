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

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams.update({'font.size': 18})


from utils import get_results, JhaPreprocessor
from train_models import test


DATAPATH = '../data/data/'

constructs = ['sentiment', 'sexism', 'hatespeech']
test_types = ['in-domain',
              'out-domain',
             ]

adv_test_types = ['adv_inv',
               'adv_swap'
                 ]
runs = 5

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

labels = {'sentiment' : {'positive': 1, 'negative' : 0},
          'sexism' : {'sexist' : 1, 'non-sexist' : 0},
          'hatespeech' : {'hate' : 1, 'not hate' : 0}
                          }

models = {
			'logreg' : {'pipeline' : Pipeline([
                        ('preprocess', JhaPreprocessor()),
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
          'bert' : {'params' : {
                       'epochs':[3, 4],
                       'learning_rate':[2e-5, 3e-5, 5e-5]
                              }
                  }
                }


# different from the train function in train_models.py since it explicitly takes in counterfactual_proportion
def train(construct = 'sentiment', counterfactual = False,
          preprocessing = None, model_name = 'logreg',
          labels = {'positive' : 1, 'negative' : 0}, one_sided = False,
          counterfactual_proportion = 0.5):
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
            non_sexist_ = non_sexist.sample(n = int(len(non_sexist)*(1-counterfactual_proportion)))
            # replace with that much counterfactual
            extra = pd.read_csv(DATAPATH+training_data[construct]['counterfactual'],
                                sep = '\t').sample(n = int(len(non_sexist)*counterfactual_proportion))
            train_data = train_data[train_data['sexism'] == 'sexist'] # take full sexist data
            train_data = train_data.append(non_sexist_) # add original non-sexist data (1 - cf_prop) %
            train_data = train_data.append(extra) # add counterfactual data of cf_prop %
            
        else:
            train_data_ = train_data.sample(n = int(len(train_data)*(1-counterfactual_proportion)))
            extra = pd.read_csv(DATAPATH+training_data[construct]['counterfactual'],
                                sep = '\t').sample(n = int(len(train_data)*counterfactual_proportion))
            train_data = train_data_.append(extra)
                
        
        
    train_data[construct] = train_data[construct].map(labels)
    print(counterfactual, len(train_data), train_data[construct].unique())
    
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


trained_models = {}
proportions = range(0, 6)
multiplier = 0.2

for construct in constructs:
    trained_models[construct] = {}
    for run in range(runs):
        for model_name in models.keys():
            for mode in [True]: #change and add counterfactuals as well
                if model_name not in trained_models[construct]:
                    trained_models[construct][model_name] = {True: {}, False: {}}
                for prop in proportions: # prop == 0 means the model is essentially non-counterfactual
                    actual_prop = prop * multiplier 
                    if actual_prop not in trained_models[construct][model_name][mode].keys():
                        trained_models[construct][model_name][mode][actual_prop] = []
                    if construct == 'sexism':
                        one_sided = True
                    else:
                        one_sided = False
                    
                    try:
                        trained_models[construct][model_name][mode][actual_prop].append(train(construct, 
                                                              model_name = model_name,
                                                              counterfactual = mode,
                                                              labels = labels[construct],
                                                              one_sided = one_sided,
                                                              counterfactual_proportion = actual_prop))
                    except Exception as e:
                        print(e)
                        trained_models[construct][model_name][mode][actual_prop].append('not enough counterfactual data')
                        pass



# construct ---> model ---> prop ---> test_set
prop_results = {}
all_results = []

for construct in constructs:
    for model in models:
        for prop in proportions:
            actual_prop = multiplier * prop
            for test_type, test_path in test_data[construct].items():
                test_set = pd.read_csv(DATAPATH + test_path, sep = "\t")
                for run in range(runs):
                    #print(test_set)
                    print(construct, model, actual_prop, test_type)
                    if trained_models[construct][model][True][actual_prop][0] != 'not enough counterfactual data':
                        true, pred, cr = test(trained_models[construct][model][True][actual_prop][run],
                                          test_set, construct, test = test_type, labels = labels[construct])
                        all_results.append(get_results(cr, true, pred,
                                              method = model,
                                              mode = 'True_%f' %(actual_prop),
                                              construct = construct,     
                                              labels = {str(v): k for k, v in labels[construct].items()}, #invert label mapping
                                              dataset = test_type))

results = {}
result_df = pd.DataFrame(all_results)


fig, ax = plt.subplots(1, 2, figsize = (20, 5), sharey = True, sharex = False)
for n, test_type in enumerate(test_types):
    result_df_ = result_df[result_df['dataset'] == test_type]
    result_df_['counterfactual proportion'] = [float(i[5:9]) for i in result_df_['mode']]
    if n == 1:
        legend_ = False
    else:
        legend_ = True
    sns.lineplot(data = result_df_, x = 'counterfactual proportion', y = 'Macro F1',
                 hue = 'construct', err_style = 'bars', ax = ax[n], legend = legend_)
    ax[n].set_title("Performance on %s" %(test_type))
    
plt.savefig("../results/plots/injection_proportion.pdf", bbox_inches='tight')                                                                      
