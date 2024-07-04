# From _Showgirls_ to _Performers_: Fine-tuning with Gender-inclusive Language for Bias Reduction in LLMs

This repository contains the official code and data for the experiments carried out in our paper "From _Showgirls_ to _Performers_: Fine-tuning with Gender-inclusive Language for Bias Reduction in LLMs". 

The steps are as follows: 

1. Download dataset 
2. Rewrite dataset with gender-neutral terminology
4. Fine-tune LLMs (GPT-2, Phi-1.5, RoBERTa) with rewritten vs. original data
5. Evaluate with external metrics (RedditBias, CrowsPairs, HONEST)


Create `external_libs` directory and clone the following repositories: 

1. [NeuTralRewriter](https://github.com/vnmssnhv/NeuTralRewriter)
2. [RedditBias](https://github.com/SoumyaBarikeri/RedditBias)
3. [bias-bench](https://github.com/McGill-NLP/bias-bench)