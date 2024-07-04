from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import time
from tqdm import tqdm
import argparse

if __name__ == '__main__':

    # set up argparse
    parser = argparse.ArgumentParser(description='get number of examples to evaluate on')
    parser.add_argument('--k', help='path to model directory', required=True)
    parser.add_argument('--log_dir', help='path to logging directory', required=True)

    args = parser.parse_args()

    # honest

    lang='en'
    k = int(args.k)
    family = 'causal-lm'

    import honest.honest as honest

    phi_prefix = 'phi-1_5-fine-tuned-'

    model_names = [phi_prefix+'3epoch-neutral', phi_prefix+'3epoch-R',
                   phi_prefix+'1epoch-neutral-R', phi_prefix+'1epoch-R']

    model_names = ['microsoft/phi-1_5']

    for m in model_names:
        print(f'MODEL: {m}')
        name_model = f'models/llms/{m}'
        name_model = m

        # Load HONEST templates
        evaluator = honest.HonestEvaluator(lang)
        for mode in ['binary','queer_nonqueer']:#,'all']:
            print(f'honest mode: {mode}')
            st = time.time()

            masked_templates = evaluator.templates(data_set=mode) # "binary" or "queer_nonqueer" or "all"

            # Load BERT model
            tokenizer = AutoTokenizer.from_pretrained(name_model)

            if 'bert' in m:
                model = AutoModelForMaskedLM.from_pretrained(name_model)
            else:
                model = AutoModelForCausalLM.from_pretrained(name_model)

            
            if family == 'masked-lm':

                # Define nlp_fill pipeline
                nlp_fill = pipeline('fill-mask', model=model, tokenizer=tokenizer, top_k=k)

                # print("FILL EXAMPLE:",nlp_fill('all women like to [M].'.replace('[M]',tokenizer.mask_token)))

                # Fill templates (please check if the filled words contain any special character)
                filled_templates = [[fill['token_str'].strip() for fill in nlp_fill(masked_sentence.replace('[M]',tokenizer.mask_token))] for masked_sentence in masked_templates.keys()]

                honest_score = evaluator.honest(filled_templates, masked_templates)
            
            elif family == 'causal-lm':
                generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
                set_seed(42)

                filled_templates = []

                masked_template_keys = list(masked_templates.keys())

                for i in tqdm(range(len(masked_template_keys))):

                    masked_sentence = masked_template_keys[i]

                    fills = generator(masked_sentence.replace(' [M].',''),
                                      max_length=len(tokenizer(masked_sentence.replace(' [M].',''))['input_ids'])+10,
                                      num_return_sequences=k,
                                      pad_token_id=generator.tokenizer.eos_token_id,
                                      num_beams=k
                                      )

                    for fill in fills:
                        filled_list = [fill['generated_text'][len(masked_sentence.replace(' [M]','')):]]
                        filled_templates.append(filled_list)
                
                #Compute HONEST score
                honest_score, honest_df = evaluator.honest_dataframe(filled_templates, masked_templates)

            print(name_model, k, round(honest_score, 3), mode)
            with open(args.log_dir + f'/honest_scores.txt', 'a+') as f:
                f.write(f'model: {name_model}\nk: {k}\nhonest_score: {round(honest_score, 3)}\nmode: {mode}\n\n\n') 

            et = time.time()
            # print out nicely formatted time
            print(f'time elapsed: {round((et-st)/60, 1)} minutes')