import argparse
import os
import pandas as pd
import random

from tqdm import tqdm

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Get model directory and path to files that need to be rewritten.')
    parser.add_argument('--corpus', help='path to corpus directory', required=True)
    parser.add_argument('--new_path', help='provide a new name for the filtered corpus')

    args = parser.parse_args()

    corpus_path = args.corpus
    if corpus_path[-1] == '/':  # remove slash from end of path
        corpus_path = corpus_path[:-1]

    if not args.new_path:
        corpus_path_fil = corpus_path + '-R'
    else:
        corpus_path_fil = args.new_path

    print(f'path for new corpus: {corpus_path_fil}')

    if not os.path.exists(corpus_path_fil):
        os.makedirs(corpus_path_fil)

    # read in replacement dictionary
    replacement_df = pd.read_csv('replacements+plural-final.csv')
    added_replacements = pd.read_csv('data/terms/gender_neutral_lexicon_vanmassenhove.csv')

    #remove all rows in added_replacements that are already in replacement_df
    added_replacements = added_replacements[~added_replacements['word'].isin(replacement_df['word'].values)]
    # concatenate both dataframes
    replacement_df = pd.concat([replacement_df, added_replacements], ignore_index=True)


    words_to_find = set(replacement_df['word'].values)
    word_counter = 0

    for sub_corpus in os.listdir(corpus_path):
        print(f'processing {sub_corpus} now')

        # create subcorpus directory in the new neutral corpus path
        if not os.path.exists(os.path.join(corpus_path_fil, sub_corpus)):
            os.makedirs(os.path.join(corpus_path_fil, sub_corpus))

        # loop through all the files
        fnames = os.listdir(corpus_path + '/' + sub_corpus)
        for i in tqdm(range(len(fnames)), desc=sub_corpus): # show progress bar
            fname = fnames[i] # current file

            with open(corpus_path + '/' + sub_corpus + '/' + fname, mode='r') as in_file, \
                    open(corpus_path_fil + '/' + sub_corpus + '/' + fname, mode='w') as out_file:
                for line in in_file:
                    line = line.strip().split()  # import tokenized line
                    # remove all gender-specific pronouns
                    new_line = []
                    for w in line:
                        if w.lower() in words_to_find:
                            word_counter += 1
                            replacement = replacement_df[replacement_df['word'] == w.lower()]['replacement'].values[0]
                            # if there is more than one replacement, pick one at random
                            if ',' in replacement:
                                replacement = replacement.split(', ')
                                replacement = random.choice(replacement)

                            # capitalize if the original word was capitalized
                            if w[0].isupper():
                                new_line.append(replacement.capitalize())
                            else:
                                new_line.append(replacement)
                        else:
                            new_line.append(w)
                    # new_line = [w for w in line if w.lower() not in pros]
                    out_file.write(' '.join(new_line) + '\n')

    print(f'{word_counter} words were replaced in entire corpus')
