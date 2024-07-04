import argparse
import os
import sys

from tqdm import tqdm

from utils import Timer, timestamp, logging

sys.path.append("external_libs/NeuTralRewriter/")
from rewrite_neutral import NeutralRewriter

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Provide the path to corpus files that need to be rewritten."
    )
    parser.add_argument("--corpus", help="path to training corpus directory", required=True)
    parser.add_argument(
        "--log", help="path to log file. If not provided, will create and write to logs directory."
    )

    args = parser.parse_args()

    nr = NeutralRewriter(
        language="en", parse=True, advanced=True
    )  # advanced = True uses list of neutral words

    #### LOGGING ####

    if args.log:
        if len(args.log.split(".")) == 2:
            log_file = args.log
        elif args.log[-1] == "/":
            log_file = args.log + "rewriting_log_" + timestamp() + ".txt"
        else:
            log_file = args.log + "/rewriting_log_" + timestamp() + ".txt"
    else:
        if not os.path.exists("logs/"):
            os.makedirs("logs/")
        log_file = "logs/rewriting_log_" + timestamp() + ".txt"

    t = Timer()
    t.start()
    logging("starttime: " + timestamp(), log_file)

    #####

    corpus_path = args.corpus if args.corpus[-1] != "/" else args.corpus[:-1]
    corpus_path_neutral = corpus_path + "-neutral"
    corpus_path_tok = corpus_path + "-tok"

    existing_files = []

    for sub_corpus in os.listdir(corpus_path):

        # create subcorpus directory in the new neutral corpus path
        if not os.path.exists(os.path.join(corpus_path_neutral, sub_corpus)):
            os.makedirs(os.path.join(corpus_path_neutral, sub_corpus))
        if not os.path.exists(os.path.join(corpus_path_tok, sub_corpus)):
            os.makedirs(os.path.join(corpus_path_tok, sub_corpus))
        else:
            existing_files += os.listdir(os.path.join(corpus_path_neutral, sub_corpus))

        # loop through all the files
        fnames = os.listdir(os.path.join(corpus_path, sub_corpus))

        for i in tqdm(range(len(fnames)), desc=sub_corpus):
            fname = fnames[i]
            if fname in existing_files:
                # print(f"{fname} already rewritten")
                logging(f"{fname} already rewritten", log_file)
            else:
                logging(f"processing {fname}", log_file)
                with open(os.path.join(corpus_path, sub_corpus, fname), mode="r") as in_file, open(
                    os.path.join(corpus_path_neutral, sub_corpus, fname), mode="w"
                ) as out_file, open(
                    os.path.join(corpus_path_tok, sub_corpus, fname), mode="w"
                    ) as out_file_tok:
                    for line in in_file:
                        # convert to unicode if not already
                        line = bytes(line, "utf-8").decode("utf-8", "ignore")
                        neutral_sent, tok_sent = nr.process_sentence(
                            line.strip(), parse=True
                        )  # rule-based; only take the neutral version
                        # print(neutral_sent)
                        out_file.write(f"{neutral_sent}\n")
                        out_file_tok.write(f"{tok_sent}\n")


    print(t.stop())
    logging("endtime:" + timestamp(), log_file)
