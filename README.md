Machine Learning Operations
Exam Project: Group 27
==============================
This GitHub Repository contains the main components of our exam project. The group consists of the following students from the Technical University of Denmark: 
- Jonas Poulsen (s194243)
- Andreas Hornemann Nielsen (s194236)
- Christian Vestergaard Djurhuus (s194244)
- Xiang Bai (s213120)

Using the repository
------------
A brief giude of how to install and use the repository.

# Using the repository locally and run Dockerfiles
- Clone the repository
- Navigate into the cloned repository
- ```pip install -r requirements.txt```
- ```pip install -e .```

# Using Google Cloud Project (GCP) and make it run with GitHub Actions
First of, we recommend that you fork the repository into your own GitHub account. By doing this, you will be able to use the GitHub actions later on to automatically push Dockefiles to your GCP. After having forked the repository
- GCP Setup
- Setting up GitHub Secrets

Project description
------------
The purpose of the following project is to become acquainted with the production machine learning life (ML) cycle (Design, Model development and Operations) with a particular focus on the operation stage. Thus, the primary goal of the project is to learn how to manage a production ML life cycle through the usage of good practices and the tools presented in the course “Machine Learning Operations - 02476”.


The model used for this project is BERT (Bidirectional Encoder Representations from Transformers) published by Google AI Language and is a part of the Transformer framework built by the Huggingface group. 

The motivation for using BERT is that, despite its simplicity, it is a very powerful tool that has reached state-of-the-art results on several NLP tasks. Furthermore, it supports PyTorch which is in line with what is used in the course. Therefore, BERT fits perfectly into the goal of the project. To use machine learning operation tools - not designing cool AI models. 

The main task of our model is to perform binary sentiment classification using text on the IMDb dataset. For the training, a fine-tuning method approach has been chosen. It is a wise trade-off， considering more time can be devoted to the use of the tools that our course provides. The model will be fine-tuned on a labelled sub-dataset after it having been pre-trained on a large unlabelled dataset to achieve the effect of training the model faster. There is a Trainer API in the Transformers library, which allows for easy logging, gradient accumulation, mixed precision and some evaluations for the training.


The dataset used for the sentiment project is the following from hugging face:
“ https://huggingface.co/datasets/imdb ”
The IMDB dataset consists of 100.000 plain text comments regarding movies. 50.000 of which are labelled as a binary dataset using the label of either “neg” or “pos”. The other 50.000 data points are however unlabelled. Initially, the 50.000 labelled data points will be used for training and testing, however, the unlabelled set may be used for potential further pretraining, if seen fit.



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── utils          <- util files such as deployment etc.  
    │   │
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Project checklist

Please note that all the lists are *exhaustive* meaning that I do not expect you to have completed every
point on the checklist for the exam.

### Week 1

- [x] Create a git repository
- [x] Make sure that all team members have write access to the github repository
- [x] Create a dedicated environment for you project to keep track of your packages (using conda)
- [x] Create the initial file structure using cookiecutter
- [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and 
- [x] Add a model file and a training script and get that running (and evaluation)
- [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
- [x] Remember to comply with good coding practices (`pep8`) while doing the project
- [x] Do a bit of code typing and remember to document essential parts of your code
- [x] Setup version control for your data or part of your data
- [x] Construct one or multiple docker files for your code
- [x] Build the docker files locally and make sure they work as intended
- [x] Write one or multiple configurations files for your experiments
- [x] Used Hydra to load the configurations and manage your hyperparameters
- [ ] When you have something that works somewhat, remember at some point to to some profiling and see if you can optimize your code
- [x] Use wandb to log training progress and other important metrics/artifacts in your code
- [x] Use pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

- [x] Write unit tests related to the data part of your code
- [x] Write unit tests related to model construction
- [x] Calculate the coverage. ``` currently 69%```
- [x] Get some continues integration running on the github repository
- [x] (optional) Create a new project on `gcp` and invite all group members to it
- [x] Create a data storage on `gcp` for you data
- [x] Create a trigger workflow for automatically building your docker images
- [x] Get your model training on `gcp`
- [x] Play around with distributed data loading
- [x] (optional) Play around with distributed model training
- [ ] Play around with quantization and compilation for you trained models

### Week 3

- [x] Deployed your model locally using TorchServe (TorchServe not compatible with huggingface transformers yet. Hence, we used a different approach)
- [ ] Checked how robust your model is towards data drifting
- [x] Deployed your model using `gcp`
- [x] Monitored the system of your deployed model
- [x] Monitored the performance of your deployed model

### Additional

- [x] Revisit your initial project description. Did the project turn out as you wanted?
- [x] Make sure all group members have a understanding about all parts of the project
- [x] Create a presentation explaining your project
- [x] Uploaded all your code to github
- [ ] (extra) Implemented pre-commit hooks for your project repository
- [ ] (extra) Used Optuna to run hyperparameter optimization on your model
