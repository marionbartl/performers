# From _Showgirls_ to _Performers_: Fine-tuning with Gender-inclusive Language for Bias Reduction in LLMs

This repository contains the official code and data for the experiments carried out in our paper "From _Showgirls_ to _Performers_: Fine-tuning with Gender-inclusive Language for Bias Reduction in LLMs". 

The steps are as follows: 

1. Download dataset 
2. Rewrite dataset with gender-neutral terminology
4. Fine-tune LLMs (GPT-2, Phi-1.5, RoBERTa) with rewritten vs. original data
5. Evaluate with external metrics (RedditBias, CrowsPairs, HONEST)

## Step 0

Create `external_libs` directory and clone the following repositories: 

1. [NeuTralRewriter](https://github.com/vnmssnhv/NeuTralRewriter)
2. [RedditBias](https://github.com/SoumyaBarikeri/RedditBias)
3. [bias-bench](https://github.com/McGill-NLP/bias-bench)


Then, install requirements, preferably into a virtual environment: 
```sh
pip install -r requirements.txt
```

## Step 1 &ndash; Download Corpus

Run the following code, to download the **Small Heap Corpus**, consisiting of 250M tokens. The code will simultaneously also use Vanmassenhove et al.'s (2021) NeuTral Rewriter to create a version of the corpus with _he_/_she_ pronouns replaced by singular _they_.

The original and neutral corpus will be saved as `small_heap_[# tokens](-neutral)` in the `data/` directory.
The code also creates a `logs` directory to save the progress and processing time of the download process.

```sh
python code/dataset_download.py --no_tokens 250000000 --log_dir logs/
```

The `--no_tokens` argument can be used to adjust the size of the downloaded dataset.


## Step 2 &ndash; Replace Gender-marking Words

This script will do replacement of gender-marking with gender-neutral words based on the catalogue developed at this repository: [github.com/marionbartl/affixed_words](https://github.com/marionbartl/affixed_words)

The script works on the original and neutral version of the corpus simultaneously. After replacement, the corpus directory name will have an attached '-R'. 

```sh
python code/word_replacement.py --corpus data/small_heap_50M
```

## Step 3 &ndash; Fine-tuning LLMs

### With Python Script

```
python code/fine_tune.py --model_name [huggingface model identifier] --data data/fine-tuning/tiny_heap-neutral.txt
```

### With Notebook 

For our experiments, we ran `fine_tuning.ipynb` on Google Colab. 

Fine-tuned models were not included, because they were too large for this repository. 

## Step 4 &ndash; Evaluation

### CrowS Pairs

We used Meade et al.'s (2022) [https://github.com/McGill-NLP/bias-bench](implementation) of CrowsPairs.

```sh
mkdir external_libs
cd external_libs
git clone https://github.com/McGill-NLP/bias-bench.git
```

### RedditBias

We used Barikeri et al.'s (2021) [https://github.com/umanlp/RedditBias](implementation) of RedditBias. 

```sh
cd external_libs
git clone https://github.com/umanlp/RedditBias.git
```

### HONEST

For the HONEST evaluation, we used the python package from [https://github.com/MilaNLProc/honest](MilaNLP). 

The code can be found at `code/HONEST_eval.ipynb`. 

