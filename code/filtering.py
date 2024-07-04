import argparse
import os

from tqdm import tqdm

if __name__ == '__main__':
    # this code filters sentences in a tokenized corpus and kicks out those that are longer than 1000 tokens
    # this was done, because the neural models I'm working with can only handle sentences up to 1024 tokens
    # the new corpus is saved in a new directory with the extension -filtered

    parser = argparse.ArgumentParser(description='Get model directory and path to files that need to be rewritten.')
    parser.add_argument('--corpus', help='path to corpus directory', required=True)
    parser.add_argument('--max_len', help='maximum sentence length', default=1000)
    parser.add_argument('--new_path', help='provide a new name for the filtered corpus')

    args = parser.parse_args()

    max_len = int(args.max_len)

    corpus_path = args.corpus
    if corpus_path[-1] == '/':  # remove slash from end of path
        corpus_path = corpus_path[:-1]

    if not args.new_path:
        corpus_path_fil = corpus_path + '-ff'
    else:
        corpus_path_fil = args.new_path

    if not os.path.exists(corpus_path_fil):
        os.makedirs(corpus_path_fil)

    for sub_corpus in os.listdir(corpus_path):
        # print(f'processing {sub_corpus} now')

        # create subcorpus directory in the new neutral corpus path
        if not os.path.exists(corpus_path_fil + '/' + sub_corpus):
            os.makedirs(corpus_path_fil + '/' + sub_corpus)

        # loop through all the files
        fnames = os.listdir(corpus_path + '/' + sub_corpus)
        for i in tqdm(range(len(fnames)), desc=sub_corpus):
            fname = fnames[i]

            with open(corpus_path + '/' + sub_corpus + '/' + fname, mode='r') as in_file, \
                    open(corpus_path_fil + '/' + sub_corpus + '/' + fname, mode='w') as out_file:
                for line in in_file:
                    line = line.strip().split()  # import tokenized line
                    # if line is too long (can't be a proper sentence & can't be processed (limit of 1024): delete)
                    if len(line) < max_len:
                        out_file.write(' '.join(line) + '\n')
