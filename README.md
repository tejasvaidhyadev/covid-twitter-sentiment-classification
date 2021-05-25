## covid-twitter-sentiment-classification-
Sentiment classifer for Covid-tweets: Finetuned [Covid-Twitter Bert](https://arxiv.org/abs/2005.07503).

This Repo contains

- Code for Inference model.
- Trained models (Finetuned model).
- Dependencies and steps to replicate results.
- cli tool for error Analysis

## Dependencies and setup
| Dependency | Version | Installation Command |
| ---------- | ------- | -------------------- |
| Python     | 3.8     | `conda create --name covid_entities python=3.8` and `conda activate covid_entities` |
| PyTorch, cudatoolkit    | 1.5.0, 10.1   | `conda install pytorch==1.5.0 cudatoolkit=10.1 -c pytorch` |
| Transformers (Huggingface) | 2.9.0 | `pip install transformers==2.9.0` |
| Scikit-learn | 0.23.1 | `pip install scikit-learn==0.23.1` |
| scipy        | 1.5.0  | `pip install scipy==1.5.0` |
| NLTK    | 3.5  | `pip install nltk==3.5
` |
## Instructions
1. Setting up the codebase and the dependencies
     
```
 # creating conda environment
 conda env create --file env.yml
 # clone this repo
 git clone https://github.com/tejasvaidhyadev/covid-twitter-sentiment-classification.git
```

2. download the pretrained model

download pretrained model and keep it in ```pretreined_model``` folder
```
bash trained_model.sh 
```

3. Command line tool

```
python interactive.py --nolog True --trained_model path/to/trainedmodel --pretrained_dir path/pretrained_dir

```
**Example**  
The below commandline tool can be used for error analysis in BERT
```
(biasst) [tvaidhya@lo-login-02 huggingface_pytorch]$ python 

interactive.py --nolog True
loading pretraining weights
======= ======= welcome! ======== ======= 
    1. Provide input strings
    2. Input 'exit' to end

Input:Input:NEW REPORT: 85% OF PEOPLE WHO CAUGHT COVID WERE REGULAR MASK WEARERS!! So masks dont do a thing to stop the virus!! TOTAL BS!! Dems using it to CONTROL US!! #TakeOffYourMasksSheeple UNHEALTHY FOR HEALTHY PEOPLE TO WEAR MASKS!!!
NEWSOM LOOKS LIKE A TOTAL MORON
['negative']
Input:  
```
4. Inference mode

Read the data/README.md and create the csv file of desired formate

```
python inference.py -csv ./path/to/csv
```

## Miscellanous
- License- MIT