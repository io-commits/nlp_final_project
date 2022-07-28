# Ukraine - Russia conflict tweets generator

## Introduction
On that project we have created a sequence generator. 

We have used mainly tensorflow.keras and happytransformet platforms for that projext

On the sections below we will elaborate on the development and research process.

---

# Repository Contents
## You WILL find here
All of the files needed for the app to run.

## You will not find here
### binaries

---

# Repository File System
## data - blank
## experiments - blank
## model
On that folder you will find two main modules, baseline.py and gpt_neo.py.
### baseline.py
That module aggregates all the tensorflow.keras logic that pre inhabited in kaggle cloud processed notebook.
The Model Architecture is pretty much straightforward with those tasks and consists mainly of Biderectional LSTM layers.
### gpt_neo.py
That module aggregates all the huggingface related logic.
We have fine-tuned EleutherAI/gpt-neo-125M GPT-NEO model.
### commander.py
A module designed to interact with the streamlit app
### credentials.json
The mock account credentials
### evaluate.py
The module has TER, BLEU, ROUGH metrics in.
### file_manager.py
The module in charge of communicating with GDrive API to download everything locally for smooth run.
### main.py
The main file - just run it to have all going.
### main_streamlit_page.py
Has all the streamlit related logic.
### requirements
Has all the project requirements.
### token.pickle 
The occurs when processing GDriveAPI related credentials.
## Notebooks
- baseline-generator.ipynb - The code executed to train the baseline models
- exploratory.ipynb - The exploratory code
- gpt-neo.ipynb - The code executed to train gpt-neo models
- load.ipynb - The code designed to handle the data and make it pre-process friendly
- pre_process-sampled.ipynb - The preprocess related code.

---

# Research Flow
## Data
https://www.kaggle.com/datasets/bwandowando/ukraine-russian-crisis-twitter-dataset-1-2-m-rows
That is an ongoing project maintained by https://www.kaggle.com/bwandowando
The tweets are scraped from twitter using conflict dedicated scraper.
It consists currently of 47.15M tweets.
On that project we have used top of 30M.

---

## Loading the data
In order to load the data we have created a notebook that load all the gzip supplied.
Then we have merged everything to a single DataFrame.

---

## Data 'Types'

---

### Best Tweets
We have started experimenting with sorted by retweet count data.
That data had a lot of anticipated duplicates due to its nature - popular retweet highly share common ground.
While exploring that data we have seen a variety of tweets we are not mainly interested on such as "Good night !".
We have forsaken that approach due to those issues.
### Sampled Tweets
We have mainly focused on that approach. We have sampled a variety of data chucks on sizes ranges from 10K to 30M.

---

## Pre-Process
We have seen the data consists of various kinds of tokens.
From alphabetic to numeric and symbolic.
Furthermore, the Tweeter platform has some cultural habits embedded in such as hashtags and user referencing.
We have eliminated everything, to remain with more clean and concise corpus.

---

## Baseline model architecture
The model is a tensorflow.keras Sequential model.

The layers are as follows:

### Embedding
Layer the size of the total input words x vector of size 100

### BiderctionalLSTM
Layer with 150 units, which we return for further process

### Dropout 
A layer to eliminate 20% of the weights

### LSTM
A layer with 100 units trying to use previous Bidirectional influence

### Dense 
A layer the size of half of the total words, trying to shrink the weights heatmap. 

That layer has an activation function of relu, eliminating all non-positive weights.

### Dense
A layer the size of the total words, with softmax activation.

That layer actually holds the model prediction when the highest valued vector, is the highest probability prediction.

---

## GPT-NEO Architecture

That model is sequent to gpt2 and actually is an open source version to gpt3.

The model we used (due to kaggle cloud limitations) was 125M.

There are other variations online.

That model is based on transformer with attention approach.

It had been trained on a variety of TPU's on tensorflow cloud.

The model is basically a collection of models which ultimately give a rich and powerful language model.

A list of the teams publication can be found here: 

https://www.eleuther.ai/publications/

---

## Training
###Intro
We have suffered from a lot of hangups. The personal machine we poses are poor, at lease, for that task, although, despite that - been sufficient for pre-processing.
That was not the case with training so we have moved to Kaggle cloud.
### BaseLine models
The base line models had been running with [10k, 50k, 100k, 1m] chunks of data.
When the amount of epochs had been:
    
- 100 epochs for baseline1
- 300 epochs for baseline2

### gpt-neo
We have experimented with data chucks of [1m, 2m, 10m, 30m]
We have monitored the model convergence which seemed pretty sufficient for a small amount of happytransformer epochs.
Worth mentioning is the fact that we were not able to increase batch_size to more than two - it resulted on platform crash.

---

## Evaluation
We have sampled the process trying to look for semantic and syntactic coherence of the models outputs.
We have seen it progress drastically shifting from the baseline models to the gpt based models.
That is no surprise considering the rich language model gpt based models have.
We have embedded in the app TER, BLEU and ROUGH evaluations.

