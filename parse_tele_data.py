from os import listdir
from typing import List, Tuple

from lxml import etree
from os.path import join, isfile

from tqdm import tqdm


# from https://github.com/jagoleni/tele-data/blob/62de6af25b6e2276422bd51db99b460b4e37d944/tele-data.py
def parse_file(html_string):
    data = []
    root = etree.HTML(html_string)
    for element in root.iter():
        if "id" in element.attrib:
            message = {}
            message["message_id"] = element.attrib["id"]
            for child in element.getchildren():
                if (
                        element.attrib["class"] == "message service"
                        and child.attrib["class"] == "body details"
                ):
                    message["text"] = child.text.strip()
                    message["type"] = "service_message"
                if child.attrib["class"] == "body":
                    for grandchild in child.getchildren():
                        if grandchild.attrib["class"] == "from_name":
                            name = grandchild.text.strip()
                            message["name"] = name
                        if grandchild.attrib["class"] == "pull_right date details":
                            message["timestamp"] = grandchild.attrib["title"]
                        if grandchild.attrib["class"] == "text":
                            message["text"] = grandchild.text.strip()
                            message["type"] = "text"
                        if grandchild.attrib["class"] == "forwarded body":
                            message["type"] = "forwarded_message"
                        if grandchild.attrib["class"] == "media_wrap clearfix":
                            message["type"] = (
                                grandchild.getchildren()[0].attrib["class"].split()[-1]
                            )
            if element.attrib["class"] == "message default clearfix joined":
                message["joined_message"] = True
                message["name"] = name
            if element.attrib["class"] == "message default clearfix":
                message["joined_message"] = False
            data.append(message)
    return data


def load(dir: str):
    path = join("data/", dir)
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    parsed = []
    for file in files:
        with open(file, 'r') as f:
            try:
                parsed = parsed + parse_file(f.read())
            except:
                print(f"Failes to parse: {file}")
    filtered = filter(lambda x: 'text' in x and 'name' in x, parsed)
    filtered = [(message['name'], message['text']) for message in filtered]

    return convolute_messages(filtered)


def convolute_messages(messages: List[Tuple[str, str]], window=12, task=None):
    new_messages = []
    for idx, (name, text) in tqdm(enumerate(messages),total=len(messages)):
        entry = {}
        max_idx = min(len(messages), idx + window)
        if task != None:
            entry['instruction'] = task
        else:
            entry[
                'instruction'] = f"You are in the middle of a conversation between friends in a group chat. Continue the conversation, match the tone and character of the conversation."
        entry['query'] = ""
        entry['output'] = ""
        for i in range(idx, max_idx):
            if i <=max_idx//2:
                entry['query'] += messages[i][0] + ": " + messages[i][1] + "\n"
            else:
                entry['output'] += "\n" + messages[i][0] + ": " + messages[i][1]
        new_messages.append(entry)
    return new_messages
