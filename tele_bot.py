import argparse

import telebot
from transformers import LlamaTokenizer

from inference import get_response, get_model

instruction = "<unk>You are in the middle of a conversation between friends in a group chat. Continue the conversation, match the tone and character of the conversation.\n"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=False, default="decapoda-research/llama-7b-hf",
                        help='name of the model')
    parser.add_argument('-cp', '--checkpoint', type=str, required=False, default=None,
                        help='e.g. logs/checkpoint-200')
    parser.add_argument('-bt', '--bot_token', type=str, required=True,
                        help='e.g. the telegram bot token')
    args = parser.parse_args()

    model_name = args.model_name
    model_path = args.checkpoint
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = (0)
    model = get_model(model_name, model_path)

    bot_token = args.bot_token
    bot = telebot.TeleBot(bot_token)



    @bot.message_handler(commands=['spam'])
    def respond(message):
        response = get_response(tokenizer, model, message.json['text'][6:], num_beams=4, max_new_tokens=128, repetition_penalty=5.2, sample=True).replace(instruction, "")
        bot.reply_to(message, response)

    print("Starting bot")
    bot.infinity_polling()
