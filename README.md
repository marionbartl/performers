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

## Step 1 - Download Corpus

Run the following code, to download the **Small Heap Corpus**, consisiting of 250M tokens. The corpus will be saved in the `data/` directory.
The code also creates a `logs` directory to save the progress and processing time of the download process.

```sh
python code/dataset_download.py --no_tokens 250000000 --log_dir logs/
```

The `--no_tokens` argument can be used to adjust the size of the downloaded dataset. 

