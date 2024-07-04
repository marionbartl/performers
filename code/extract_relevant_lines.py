import argparse
import os
import pandas as pd

from tqdm import tqdm
from spacy.tokens import Doc
import spacy
import os
# import stanza
# from stanza import Pipeline
import random


def get_word_pairs():
    # read in replacement dictionary
    #replacement_df = pd.read_csv('data/terms/replacements+plural-final.csv')
    replacement_df = pd.read_csv('data/terms/replacements_v2.tsv', sep='\t')

    added_replacements = pd.read_csv('data/terms/gender_neutral_lexicon_vanmassenhove.csv')

    #remove all rows in added_replacements that are already in replacement_df
    added_replacements = added_replacements[~added_replacements['word'].isin(replacement_df['word'].values)]
    # concatenate both dataframes
    replacement_df = pd.concat([replacement_df, added_replacements], ignore_index=True)

    return replacement_df


def custom_tokenizer(text):
    """use spacy with already tokenized text"""
    tokens = text.strip().split() 
    return Doc(nlp.vocab, tokens)

def get_sent_with_replacements(line,nlp,words_to_find,replacement_df):
    doc = nlp(line) # import tokenized line (spacy)
    new_line = [] # neutral line with replacements
    
    for w in doc:
        if w.text.lower() in words_to_find and w.ent_iob_ == 'O':
            # print('found one!', w.text.lower())
            relevant_line = True
            replacement = replacement_df[replacement_df['word'] == w.text.lower()]['replacement'].values[0]
            # capitalize if the original word was capitalized
            if w.text[0].isupper():
                new_line.append(replacement.capitalize())
            elif w.text.isupper():
                new_line.append(replacement.upper())
            else:
                new_line.append(replacement_df[replacement_df['word'] == w.text.lower()]['replacement'].values[0])
        else:
            new_line.append(w.text)
    
    return ' '.join(new_line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get model directory and path to files that need to be rewritten.')
    parser.add_argument('--norm_corpus', help='path to corpus directory', required=True)
    parser.add_argument('--neutral_corpus', help='path to neutralized corpus directory', required=True)
    parser.add_argument('--new_path', help='provide a new path to the filtered corpus', required=True)

    args = parser.parse_args()

    normal_corpus_dir = args.norm_corpus
    neutral_corpus_dir = args.neutral_corpus

    if not os.path.exists(args.new_path):
        os.makedirs(args.new_path)

    normal_lines_file = os.path.join(args.new_path,'tiny_heap.txt')
    neutral_lines_file_r = os.path.join(args.new_path,'tiny_heap-neutral-R.txt')
    neutral_lines_file = os.path.join(args.new_path,'tiny_heap-neutral.txt')
    normal_lines_file_r = os.path.join(args.new_path,'tiny_heap-R.txt')

    print(f'save reduced file here: {normal_lines_file} and {neutral_lines_file_r}')

    # import word pairs for replacement
    replacement_df = get_word_pairs()
    words_to_find = set(replacement_df['word'].values)

    print('Finding {0} terms to replace'.format(len(words_to_find)))

    # # load spacy model + custom tokenizer
    nlp = spacy.load('en_core_web_sm', disable=["parser"])
    nlp.tokenizer = custom_tokenizer

    # # load stanza model
    # nlp = Pipeline(lang='en', processors='ner,tokenize', tokenize_pretokenized=True)

    occurrence_counter = {'cc-news':0, 'openwebtext':0, 'wikipedia':0}
    extracted_df = pd.DataFrame(columns=['subcorpus', 'line', 'line_r','line_neut','line_neut_r'])

    for sub_corpus in os.listdir(neutral_corpus_dir):

        sub_corpus_norm = sub_corpus[:-8] # remove '-neutral' from subcorpus name

        print(f'processing {sub_corpus} now')

        # loop through all the files
        fnames = os.listdir(neutral_corpus_dir + '/' + sub_corpus)

        for i in tqdm(range(len(fnames)), desc=sub_corpus): # show progress bar
            fname = fnames[i] # current file

            with open(os.path.join(neutral_corpus_dir, sub_corpus, fname), mode='r') as in_file_neut, \
                    open(os.path.join(normal_corpus_dir, sub_corpus_norm, fname), mode='r') as in_file_norm, \
                        open(neutral_lines_file_r, mode='a+') as out_file_neut_r, \
                            open(normal_lines_file, mode='a+') as out_file_norm, \
                                open(neutral_lines_file, mode='a+') as out_file_neut, \
                                    open(normal_lines_file_r, mode='a+') as out_file_norm_r:
                
                # get all the normal and neutralized lines
                neutral_lines = in_file_neut.readlines()
                normal_lines = in_file_norm.readlines()

                assert len(neutral_lines) == len(normal_lines)

                for i, line in enumerate(neutral_lines):
                    # line = line.strip().split()  # import tokenized line
                    doc = nlp(line) # import tokenized line (spacy)
                    # doc = nlp(line.strip().split()) # import tokenized line (stanza)
                    relevant_line = False # flag to check if line contains a word that needs to be replaced
                    new_line = [] # neutral line with replacements

                    # # example
                    # d = nlp('Sarah Marshall ran the Boston Marathon')
                    # print([w.ent_iob_ for w in d])
                    # ents = [(e.text, e.start_char, e.end_char, e.label_) for e in d.ents]
                    # print(ents)
                    
                    for w in doc:
                        if w.text.lower() in words_to_find and w.ent_iob_ == 'O':
                            # print('found one!', w.text.lower())
                            relevant_line = True
                            replacement = replacement_df[replacement_df['word'] == w.text.lower()]['replacement'].values[0]
                            if ',' in replacement:
                                replacement = replacement.split(', ')
                                replacement = random.choice(replacement)
                            # capitalize if the original word was capitalized
                            if w.text[0].isupper():
                                new_line.append(replacement.capitalize())
                            elif w.text.isupper():
                                new_line.append(replacement.upper())
                            else:
                                new_line.append(replacement)
                        else:
                            new_line.append(w.text)
                    
                    # only get lines that contain word replacements
                    if relevant_line: 
                        occurrence_counter[sub_corpus_norm] += 1
                        out_file_neut_r.write(' '.join(new_line) + '\n') # neutral - R
                        out_file_norm.write(normal_lines[i]) # normal
                        out_file_neut.write(neutral_lines[i]) # neutral
                        sent_r = get_sent_with_replacements(normal_lines[i],nlp,words_to_find,replacement_df)
                        out_file_norm_r.write(sent_r+'\n') # normal - R
                        extracted_df = extracted_df._append({'subcorpus':sub_corpus_norm, 
                                                            'line':normal_lines[i].strip(), 
                                                            'line_r':sent_r, 
                                                            'line_neut':neutral_lines[i].strip(), 
                                                            'line_neut_r':' '.join(new_line)}, ignore_index=True)

    print(f'# words replaced per subcorpus: {occurrence_counter}')
    print(f'# words replaced: {sum(occurrence_counter.values())}')

    # save dataframe
    extracted_df.to_csv(os.path.join(args.new_path,'tiny_heap_variants.csv'), index=False)
