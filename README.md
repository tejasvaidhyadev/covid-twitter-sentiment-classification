## COVID Twitter Sentiment Classifier

This repo contains our fine-tuned **BERT sentiment classifier** for **COVID tweets**. Our model adopts and finetunes the [Covid-Twitter Bert model](https://arxiv.org/abs/2005.07503) (Müller et al., 2020).



You can easily use our **off-the-shelf** model by a simple **1-line command**.



Main author of this repo: _Tejas Vaidhya_. 
This is part of the project on NLP for Policy-Making led by _Zhijing Jin (Max Planck Institute), Zeyu Peng (MIT), Tejas Vaidhya (IIT), Bernhard Schoelkopf (Max Planck Institute), and Rada Mihalcea (University of Michigan)_.


**In this repo, we provide the following resources:**

- Our finetuned BERT model (and model weights saved [here](https://github.com/tejasvaidhyadev/covid-twitter-sentiment-classification/releases/download/V0.1/weights.zip))
- Command to run our BERT model (with both interactive and inference modes)
- Dependencies and step-by-step guides

  

## Directory Structure

- **data**: Contains pre-processed csv file of data. Read data/README.md for more details.  

- **experiment**: Contains experiments related log and models  

- **pretrained_model**: Pretrained models and config jsons  

- **env.yml**: yaml file to create conda environment

- **inference.py**: code for runing model in inference mode

- **interactive.py**: code to run model in interactive mode

- **README**: This file :)

- **train.py**: code for training the Covid twitter bert

- **util.py**: contains utilty function

  

## Instructions

### 1. Set up the codebase and the dependencies

```bash
# creating conda environment
conda env create --file env.yml
# clone this repo
git clone https://github.com/tejasvaidhyadev/covid-twitter-sentiment-classification.git
```
Below are the dependencies of our model:

| Dependency                 | Version     | Installation Command                                         |
| -------------------------- | ----------- | ------------------------------------------------------------ |
| Python                     | 3.8         | `conda create --name covid_entities python=3.8` and `conda activate covid_entities` |
| PyTorch, cudatoolkit       | 1.5.0, 10.1 | `conda install pytorch==1.5.0 cudatoolkit=10.1 -c pytorch`   |
| Transformers (Huggingface) | 2.9.0       | `pip install transformers==2.9.0`                            |
| Scikit-learn               | 0.23.1      | `pip install scikit-learn==0.23.1`                           |
| scipy                      | 1.5.0       | `pip install scipy==1.5.0`                                   |
| NLTK                       | 3.5         | `pip install nltk==3.5<br/>`                                 |



### 2. Download the pretrained model

Download [pretrained model](https://github.com/tejasvaidhyadev/covid-twitter-sentiment-classification/releases/download/V0.1/weights.zip) and keep it in ```pretrained_model``` folder

```bash
wget https://github.com/tejasvaidhyadev/covid-twitter-sentiment-classification/releases/download/V0.1/weights.zip
unzip weights.zip -d pretrained_model #make sure to del already existed JSON.  
```

### 3. Run Our Model by a Simple One-Line Command

```bash
python interactive.py --nolog True --trained_model ${path_to_trained_model} --pretrained_dir ${path_to_pretrained_model}
```

**Example to run our model in the interactive mode:**

```bash
$ python 

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

### 4. Inference mode

Read the data/README.md and create the csv file of desired formate

```bash
python inference.py -csv ${path_to_csv}
```

## Miscellanous

- This project is MIT-Licenced.
- To request new features, please start a pull request.
