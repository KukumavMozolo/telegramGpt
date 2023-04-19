import argparse
from os.path import join, exists

import torch
from peft import PeftModel, set_peft_model_state_dict, get_peft_model
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig

from train import config
import gradio as gr
import re

device = 'cuda'


def get_prompt(text: str, tokenizer):
    text = re.sub(',\s\[[0-9]*(.[0-9]*){2}\s[0-9]*:[0-9]*\]\n', ':', text)
    task = "You are in the middle of a conversation between friends in a group chat. Continue the conversation, match the tone and character of the conversation."
    p = task + '\n\n' + "### Instruction:\n" + text + '\n' + '### Response:\n'
    inputs = tokenizer(p, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    return input_ids


def get_response(tokenizer, model, text, num_beams, max_new_tokens, repetition_penalty, sample):
    generation_config = GenerationConfig(
        num_beams=num_beams,
        do_sample=sample,
        repetition_penalty=repetition_penalty,
    )
    input = get_prompt(text, tokenizer)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
        res = tokenizer.decode(generation_output.sequences[0])
        return res

def get_model(model_name:str, model_path:str):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = get_peft_model(model, config)

    checkpoint_name = join(model_path, "pytorch_model.bin")
    if exists(checkpoint_name):
        print(f"Loading weights from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        set_peft_model_state_dict(model, adapters_weights)
    if not exists(checkpoint_name):
        checkpoint_name = join(model_path)
        if exists(checkpoint_name):
            print(f"Loading weights from {checkpoint_name}")
            model = PeftModel.from_pretrained(
                model,
                checkpoint_name,
                torch_dtype=torch.float16,
            )
        else:
            print(f"No Model found at {model_path}")
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.half()
    model.eval()
    model = torch.compile(model)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=False, default="decapoda-research/llama-7b-hf",
                        help='name of the model')
    parser.add_argument('-cp', '--checkpoint', type=str, required=False, default=None,
                        help='e.g. logs/checkpoint-200')

    args = parser.parse_args()

    model_name = args.model_name
    model_path = args.checkpoint

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = (0)

    model = get_model(model_name, model_path)

    gr.Interface(
        fn=get_response,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Text",
                placeholder="Copy telegram messages here",
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Slider(
                minimum=1.0, maximum=20.0, step=1.1, value=1.0, label="Repetition penalty"
            ),
            gr.components.Checkbox(value=False, label="Sample")
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="Telegram-LoRA",
        description="Telegram-LoRA",
    ).queue().launch(server_name="0.0.0.0", share=False)
