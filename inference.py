import argparse
from os.path import join, exists

import gr as gr
import torch
from peft import PeftModel, set_peft_model_state_dict, get_peft_model, LoraConfig, get_peft_model_state_dict
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig

from parse_tele_data import convolute_messages
from train import device_map, lora_r, lora_alpha, lora_target_modules, lora_dropout, config
import gradio as gr
import re

device = 'cuda'


def get_prompt(text: str):
    text = re.sub(',\s\[[0-9]*(.[0-9]*){2}\s[0-9]*:[0-9]*\]\n', ':', text)
    task = "You are in the middle of a conversation between friends in a group chat. Continue the conversation, match the tone and character of the conversation."
    p = task + '\n\n' + "### Instruction:\n" + text + '\n' + '### Response:\n'
    inputs = tokenizer(p, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    return input_ids


def get_response(text, num_beams, max_new_tokens, repetition_penalty, sample):
    generation_config = GenerationConfig(
        num_beams=num_beams,
        do_sample = sample,
        repetition_penalty=repetition_penalty,
    )
    input = get_prompt(text)
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

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = get_peft_model(model, config)

    checkpoint_name = join(model_path, "pytorch_model.bin")
    if exists(checkpoint_name):
        print(f"Restarting from {checkpoint_name}")
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
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.half()  # seems to fix bugs for some users.
    model.eval()
    model = torch.compile(model)

    gr.Interface(
        fn=get_response,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Text",
                placeholder="Tonie: and than disaster strikes!!",
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
        title="🦙🌲 Telegram-LoRA",
        description="Telegram-LoRA",  # noqa: E501
    ).queue().launch(server_name="0.0.0.0", share=False)
