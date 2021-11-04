# SocialCAD
This repository contains the code for EMNLP'21 paper, 'How Does Counterfactually Augmented Data Impact Models for Social Computing Constructs?'


### to install
python requirements and resources:
```shell
pip install -r requirements.txt
``` 

### Data
- we make the type of counterfactual annotations available in the data folder with the respective ids. Please download the original datasets from their sources. 

### to run
- to train and test models (RQ1): (will automatically generate tables with model performance and store in results): ``train_models.py``
- to annotate CAD types: ``find and classify diffs.ipynb``
- to train models on different types of CAD (RQ2): ``train_by_typology.py``
- to get analyze feature weights (RQ3) and significance tests (RQ1): ``explain model predictions.ipynb``
- injection analysis (in Appendix): ``injection_percent.py``

