import os
from utils import big_num_to_string
import random

def get_random_file(processed, directory):
    random_file=random.choice(os.listdir(directory))
    if random_file not in processed:
        processed.append(random_file)
        return random_file, processed
    else:
        return get_random_file(processed, directory)



if __name__ == '__main__':

    # parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True, help='path to corpus')
    parser.add_argument('--no_tokens', type=str, required=True, help='number of tokens to reduce the corpus to')

    args = parser.parse_args()

    random.seed(42)

    # reduce corpus from 250M to 50M
    corpus_dict = {#'openwebtext': {'source': 'the_pile_openwebtext2',
                #               'size': 1.0},
                'openwebtext': {'source': 'Skylion007/openwebtext',
                                'size': 0.5},
                'wikipedia': {'source': 'olm/olm-wikipedia-20221001',
                                'size': 0.2},
                # 'common_crawl': {'source': 'snoop2head/common_crawl',
                #                  'size': 0.6},
                # 'books3': {'source': 'the_pile_books3',  # too big to handle (even a single instance)
                #            'size': 0.16},
                'cc-news': {'source': 'cc_news',
                            'size': 0.3}
                }
    
    try:
        no_tokens = int(args.no_tokens)
        # 1.6B tokens is approximately 8GB of data; the size of the original w2v training data
        human_corpus_size = big_num_to_string(args.no_tokens)
    except ValueError:
        raise ValueError('Please provide a valid number of tokens without any letters or other characters.')
    
    print(f'The new corpus size will be {human_corpus_size} tokens.')

    corpus_path_og = args.corpus+'-tok'
    corpus_path_neut = args.corpus + '-neutral-tok'
    # add new size in human-readable format to the corpus path instead of old size
    reduced_corpus_path_og = '_'.join(corpus_path_og.split('_')[:-1]) + '_' + human_corpus_size+'-tok'
    reduced_corpus_path_neut = reduced_corpus_path_og + '-neutral-tok'

    # create new directories for reduced-size corpus
    if not os.path.exists(reduced_corpus_path_og):
        os.makedirs(reduced_corpus_path_og)
    if not os.path.exists(reduced_corpus_path_neut):
        os.makedirs(reduced_corpus_path_neut)

    
    overall_token_counter = 0


    for sub_corpus in os.listdir(corpus_path_og):
        print(f'processing {sub_corpus} now')

        # create subcorpus directory in the new neutral corpus path
        if not os.path.exists(os.path.join(reduced_corpus_path_og, sub_corpus)):
            os.makedirs(os.path.join(reduced_corpus_path_og, sub_corpus))
        if not os.path.exists(os.path.join(reduced_corpus_path_neut, sub_corpus+'-neutral')):
            os.makedirs(os.path.join(reduced_corpus_path_neut, sub_corpus+'-neutral'))

        processed_files = []
        token_limit = no_tokens * corpus_dict[sub_corpus]['size']
        processed_tokens = 0

        # pick a random file that has not been chosen yet
        while (processed_tokens < token_limit) and (len(processed_files) < len(os.listdir(os.path.join(corpus_path_og, sub_corpus)))):
            fname, processed_files = get_random_file(processed_files, os.path.join(corpus_path_og, sub_corpus))

            # neutral copy-over
            with open(os.path.join(corpus_path_neut, sub_corpus+'-neutral', fname), mode='r') as in_file, \
                    open(os.path.join(reduced_corpus_path_neut, sub_corpus+'-neutral', fname), mode='w') as out_file:

                for line in in_file:
                    out_file.write(line)
            
            # original copy-over
            with open(os.path.join(corpus_path_og, sub_corpus, fname), mode='r') as in_file, \
                    open(os.path.join(reduced_corpus_path_og, sub_corpus, fname), mode='w') as out_file:

                for line in in_file:
                    out_file.write(line)
                    processed_tokens += len(line.split())
                    overall_token_counter += len(line.split())
            
        
        print(f'processed {big_num_to_string(overall_token_counter)} tokens')
