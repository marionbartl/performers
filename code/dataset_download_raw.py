import argparse
import os
import sys

import pandas as pd
from datasets import load_dataset

from utils import Timer, timestamp, file_size, text_processing, logging, big_num_to_string

sys.path.insert(0, 'external_libs/NeuTralRewriter/') # on local machine
#sys.path.insert(0, '/home/people/21200748/inclusive-terms/external_libs/NeuTralRewriter/') # on server

from rewrite_neutral import NeutralRewriter

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download a corpus and save it to a file.')
    parser.add_argument('--size', help='Determine the corpus size', required=True)
    parser.add_argument('--log_dir', help='where to save the log')

    args = parser.parse_args()

    # make sure log files can be saved
    if not args.log_dir:
        if not os.path.exists('logs/'):
            os.makedirs('logs/')
        log_file = 'logs/log_download_' + timestamp() + '.txt'
    else:
        log_file = args.log_dir + '/log_download_' + timestamp() + '.txt'

    timer = Timer()
    logging('start time: ' + timestamp(), log_file)

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
    
    # transform datasize text into bytes



    for corpus_name, info in corpus_dict.items():
        timer.start()

        # create a new directory for each sub-corpus
        corpus_path = r'data/' + corpus_name + '-' + human_corpus_size
        corpus_path_neutral = r'data/' + corpus_name + '-' + human_corpus_size + '-neutral'

        if not os.path.exists(corpus_path):
            os.makedirs(corpus_path)

        if not os.path.exists(corpus_path_neutral):
            os.makedirs(corpus_path_neutral)

        print(f'Processing {corpus_name}')

        # set up breaking condition (token limit) and path to save corpus file
        token_counter = 0
        size_limit = int(info['size'] * no_overall_tokens)
        file_number = 0
        file_path = corpus_path + '/{0}_{1:05d}.txt'.format(corpus_name, file_number)
        file_path_neutral = corpus_path_neutral + '/{0}_{1:05d}.txt'.format(corpus_name, file_number)

        # this is used to check processing time later
        stats_dict = {'tokens': [], 'seconds': []}

        dataset = load_dataset(info['source'], split='train', streaming=True)

        # randomly shuffle the dataset
        # shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10_000) 

        # loop through dataset, tokenize & save the downloaded data until enough data have been downloaded
        for i, instance in enumerate(shuffled_dataset):

            # # heavyweight text processing
            # text = instance['text']
            # data = TextData()
            # data.from_string(text)
            # data.process('stanza')  # tokenize with Stanza (possible other values: 'nltk', 'spacy')

            # Writing to file
            with open(file_path, 'a+', encoding='utf-8') as file, \
                    open(file_path_neutral, 'a+', encoding='utf-8') as neutral_file:
                
                for neutral_sent, sent in nr.process_document(instance['text']):
                    no_tokens = len(sent.split())
                    if no_tokens < 1000:

                        # make sure everything is unicode
                        sent_string = bytes(sent, 'utf-8').decode('utf-8', 'ignore')
                        neutral_sent_string = bytes(neutral_sent, 'utf-8').decode('utf-8', 'ignore')

                        # write unchanged + neutral sentence to file
                        file.write(sent_string + '\n')
                        neutral_file.write(neutral_sent_string + '\n')


            # Logging information
            if i > 0 and i % 1000 == 0:  # print information every n instances
                with open(log_file, 'a+', encoding='utf-8') as fout:
                    fout.write('{0} instances and {1} tokens processed\n'.format(i, token_counter))
                    fout.write(timer.stop() + '\n')
                # print(i, 'instances processed')
                # print('token tally: {}'.format(token_counter))
                elapsed = timer.stop(verbosity=False)
                stats_dict['tokens'].append(token_counter)
                stats_dict['seconds'].append(elapsed)

            # with data manager:
            # token_counter += data.get_metrics()  # returns the number of words in a corpus
            token_counter += no_tokens

            # keep track of file size
            fsize = file_size(file_path).split()
            size_num = float(fsize[0])
            size_cat = fsize[1]
            # if file size has reached one megabyte move over to new file
            if size_cat == 'M' and size_num >= 1:
                file_number += 1
                # path = '../data/small-pile/{0}/{0}_{1:03d}.txt'.format(corpus_name, file_number)
                file_path = corpus_path + '/{0}_{1:05d}.txt'.format(corpus_name, file_number)
                file_path_neutral = corpus_path_neutral + '/{0}_{1:05d}.txt'.format(corpus_name, file_number)

            if token_counter >= size_limit:
                break

        print(timer.stop())  # check how long processing that particular corpus took

        # check the running time in relation to the number of tokens that were processed
        df = pd.DataFrame(stats_dict)
        df.to_csv('logs/processing_time_' + corpus_name + '.csv')
        # df.plot.scatter(x='seconds', y='tokens')
        # plt.show()
    logging('end time: ' + timestamp(), log_file)

    print('finished!')
