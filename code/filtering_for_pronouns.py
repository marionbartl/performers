import argparse
import os
import stanza

from tqdm import tqdm

if __name__ == '__main__':
    # this code filters sentences in a tokenized corpus and rewrites all pronouns that weren't rewritten before
    # the new corpus is saved in a new directory with the extension -+

    parser = argparse.ArgumentParser(description='Get model directory and path to files that need to be rewritten.')
    parser.add_argument('--corpus', help='path to corpus directory', required=True)
    parser.add_argument('--new_path', help='provide a new name for the filtered corpus')

    args = parser.parse_args()

    male_pros = {'he': 0, 'him': 0, 'his': 0, 'himself': 0}
    female_pros = {'she': 0, 'her': 0, 'hers': 0, 'herself': 0}
    pros = {**male_pros, **female_pros}
    pros = set(pros.keys())

    nlp = stanza.Pipeline('en', processors='tokenize', tokenize_no_ssplit=True)

    corpus_path = args.corpus
    if corpus_path[-1] == '/':  # remove slash from end of path
        corpus_path = corpus_path[:-1]

    if not args.new_path:
        corpus_path_fil = corpus_path + '+'
    else:
        corpus_path_fil = args.new_path

    if not os.path.exists(corpus_path_fil):
        os.makedirs(corpus_path_fil)

    for sub_corpus in os.listdir(corpus_path):
        # print(f'processing {sub_corpus} now')

        # create subcorpus directory in the new neutral corpus path
        if not os.path.exists(os.path.join(corpus_path_fil, sub_corpus)):
            os.makedirs(os.path.join(corpus_path_fil, sub_corpus))

        # loop through all the files
        fnames = os.listdir(os.path.join(corpus_path, sub_corpus))
        for i in tqdm(range(len(fnames)), desc=sub_corpus):
            fname = fnames[i]

            with open(os.path.join(corpus_path, sub_corpus, fname), mode='r') as in_file, \
                    open(os.path.join(corpus_path_fil, sub_corpus, fname), mode='w') as out_file:
                for line in in_file:

                    # tokenize with stanza
                    doc = nlp(line)

                    old_line = []

                    for sentence in doc.sentences:
                        for token in sentence.tokens:
                            old_line.append(token.text)
                    
                    # line = line.strip().split()  # import tokenized line
                    # # remove all gender-specific pronouns
                    new_line = []
                    for w in line:
                        if w.lower() in pros:
                            pros[w.lower()] += 1
                        else:
                            new_line.append(w)
                    # # new_line = [w for w in line if w.lower() not in pros]
                    out_file.write(' '.join(old_line) + '\n')
    
    # print(pros)
        
