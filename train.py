from os.path import join, exists
from typing import List

import torch
import transformers
from datasets import Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, \
    set_peft_model_state_dict, get_peft_model_state_dict

import parse_tele_data
import argparse

device_map = "auto"
seed = 31337
batch_size: int = 128
micro_batch_size: int = 4
gradient_accumulation_steps = batch_size // micro_batch_size
learning_rate: float = 3e-4
cutoff_len: int = 256
lora_r: int = 8
lora_alpha: int = 16
lora_dropout: float = 0.05
lora_target_modules: List[str] = ["q_proj", "v_proj"]

config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

def train(model_name: str, data_name: str, resume_from_checkpoint: str,
          number_of_epochs: float, create_dataset: bool, fraction_of_test_data: float):

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"

    def tokenize(instruction, query, output):
        p = instruction + '\n\n' + "### Instruction:\n" + query + '\n' + '### Response:\n' + output
        result = tokenizer(
            p,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if result["input_ids"][-1] != 2:
            result["input_ids"].append(2)
            result["attention_mask"].append(1)

        if result["input_ids"][0] != 1:
            result["input_ids"][0] = 1

        result["labels"] = result["input_ids"].copy()

        return result

    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, config)

    if resume_from_checkpoint:
        checkpoint_name = join(resume_from_checkpoint, "pytorch_model.bin")
        if exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"No checkpoint found at {resume_from_checkpoint}")

    model.print_trainable_parameters()

    train_path = f'data/{data_name}-train.ds'
    test_path = f'data/{data_name}-test.ds'

    if create_dataset:
        print("Creating dataset")
        data = parse_tele_data.load(data_name)
        dataset = Dataset.from_list(data)
        dataset = dataset.map(tokenize, input_columns=['instruction', 'query', 'output']).shuffle(seed=seed)
        dataset = dataset.train_test_split(test_size=fraction_of_test_data, seed=seed)
        train_data = dataset['train']
        test_data = dataset['test']
        train_data.save_to_disk(train_path)
        test_data.save_to_disk(test_path)
    else:
        print("Using precomputed dataset")

    train_data = Dataset.load_from_disk(train_path)
    test_data = Dataset.load_from_disk(test_path)

    print("train data looks like: ")
    print(train_data[0])
    print("test data looks like: ")
    print(test_data[0])

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=test_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=number_of_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            output_dir=data_name+'-logs',
            save_total_limit=3,
            load_best_model_at_end=True,
            group_by_length=False,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    old_state_dict = model.state_dict

    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    model = torch.compile(model)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(data_name+'-model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=False, default="decapoda-research/llama-7b-hf",
                        help='name of the model')
    parser.add_argument('-d', '--data_name', type=str, required=True,
                        help='name of the folder in data dir that contains the telegram data')
    parser.add_argument('-re', '--resume_from_checkpoint', type=str, required=False, default=None,
                        help='e.g. logs/checkpoint-200')
    parser.add_argument('-e', '--number_of_epochs', type=float, default=1.0, help='number of epochs')
    parser.add_argument('-ft', '--fraction_of_test_data', type=float, required=True, default=0.001, help='fraction of data used for testing')
    parser.add_argument('--create_dataset', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    model_name = args.model_name
    data_name = args.data_name
    resume_from_checkpoint = args.resume_from_checkpoint
    number_of_epochs = args.number_of_epochs
    create_dataset = args.create_dataset
    fraction_of_test_data = args.fraction_of_test_data

    train(
        model_name,
        data_name,
        resume_from_checkpoint,
        number_of_epochs,
        create_dataset,
        fraction_of_test_data
    )
