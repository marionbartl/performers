from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np
import evaluate
import os

if __name__ == '__main__':

    # parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Get model name (currently only causal LM supported) and path to data.')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--data', type=str, help='path to data')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs')

    args = parser.parse_args()
    
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using {} device'.format(device))

    # go through the dataset with os and find the number of lines in all files combined
    data_path = args.data
    print(f'data are here {data_path}')

    if os.path.isfile(data_path):
        print('data_dir is a file')
        data_type = 'file'
        num_samples = sum(1 for line in open(data_path))
        dataset = load_dataset("text", data_files={"train": data_path}, streaming=True)
        
    else:
        print('data_dir is a directory')
        num_samples = 0
        for dirpath, dirnames, filenames in os.walk(data_path):
            for filename in filenames:
                if filename.endswith('.txt'):
                    num_samples += sum(1 for line in open(os.path.join(dirpath, filename)))
        
        # Prepare your training data
        dataset = load_dataset(data_path, streaming=True)

    print('num_samples: ', num_samples)

    # num_samples = 20
    batch_size = 2  # 8 = default
    gradient_accumulation_steps = 1  # 1 = default
    epochs = args.epochs  # 3 = default

    # from: https://stackoverflow.com/questions/76011298/huggingface-trainer-max-step-to-set-for-streaming-dataset
    max_steps = (
        num_samples // batch_size) // gradient_accumulation_steps * epochs
    print('max_steps: ', max_steps)

    # Load the pre-trained model and tokenizer
    model_name = args.model_name  # Replace with the desired model name 
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    

    def encode(examples):
        result = tokenizer(
            examples['text'], truncation=True, padding='max_length', max_length=1024)
        result['labels'] = result['input_ids'].copy()
        return result

    tokenized_dataset = dataset.map(
        encode, batched=True, batch_size=batch_size)

    small_eval_dataset = tokenized_dataset['train'].shuffle(seed=42).take(50)
    small_train_dataset = tokenized_dataset['train'].shuffle(seed=42).skip(50)

    perplexity_metric = evaluate.load('perplexity', module_type='metric')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return perplexity_metric.compute(predictions=predictions, model_id=model_name)

    training_args = TrainingArguments(
        output_dir='results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # save_steps = 10,
        logging_dir='logs',
        #logging_steps=10,
        save_strategy='steps',
        save_steps=0.3,
        save_total_limit=2,
        #load_beste_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        # data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Fine-tune the model
    train_result = trainer.train()

    # Save the model
    print(trainer.state.log_history)

    trainer.save_model('models/'+model_name+'-fine-tuned')

    metrics = train_result.metrics
    trainer.save_metrics(split='all', metrics=metrics)
