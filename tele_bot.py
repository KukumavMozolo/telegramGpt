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
    parser.add_argument('--idle', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    model_name = args.model_name
    model_path = args.checkpoint
    idle = args.idle
    if not idle:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = (0)
        model = get_model(model_name, model_path)

    bot_token = args.bot_token
    bot = telebot.TeleBot(bot_token)
    history = {}


    def add_to_history(name: str, text: str):
        if name not in history:
            history[name] = [name + ": " + text]
        else:
            if len(history[name]) > 10:
                history[name] = history[name][1:]
            history[name].append(name + ": " + text)


    def get_from_history(name: str):
        input = ""
        for h in history[name]:
            input += h + "\n"
        return input[:-1]

    @bot.message_handler(commands=['spam'])
    def respond(message):
        if not idle:
            name = message.from_user.full_name
            text = message.json['text'][5:].replace('@'+bot.get_me().username, "",1)
            add_to_history(name, text)
            input = get_from_history(name)
            response = get_response(tokenizer, model, input, num_beams=4, max_new_tokens=128, repetition_penalty=5.4,
                                    sample=True, temperature=1.0, top_p=1.00, top_k=80).replace(instruction, "")
        else:
            response = "telegramGpt: Brb, https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        print(message.json['text'][6:])
        print(response)
        bot.reply_to(message, response)

    @bot.message_handler(commands=['clear'])
    def clear(message):
        name = message.from_user.full_name
        if name in history:
            history[name] = [""]

    @bot.message_handler(commands=['repeat'])
    def repeat(message):
        if not idle:
            name = message.from_user.full_name
            input = get_from_history(name)
            response = get_response(tokenizer, model, input, num_beams=4, max_new_tokens=128, repetition_penalty=5.2,
                                    sample=True, temperature=1.0, top_p=0.75, top_k=40).replace(instruction, "")
        else:
            response = "telegramGpt: Brb, https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        print(message.json['text'][6:])
        print(response)
        bot.reply_to(message, response)


    @bot.message_handler(commands=['help'])
    def help(message):
        tutorial = """
        Tutorial for TelegramGpt:
        the commands are to be written without the <> brackets
        type </spam text> to get replies to text
        type </clear> to empty your command history 
        type </help> to get a description of the available commands 
        type </repeat> to get a new response from last input 
        """
        bot.reply_to(message, tutorial)


    print("Starting bot")
    bot.infinity_polling()
