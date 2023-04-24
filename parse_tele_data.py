from os import listdir
from typing import List, Tuple

from lxml import etree
from os.path import join, isfile
import numpy as np
from anytree import Node, RenderTree

from tqdm import tqdm

instruction = "You are in the middle of a conversation between friends in a group chat. Continue the conversation, match the tone and character of the conversation."
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
                            # message['user_id'] =
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
                        if grandchild.attrib["class"] == "reply_to details":
                            message["reply"] = grandchild.getchildren()[0].attrib['href'].split('go_to_')[1]
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
    return filtered


def convolute_messages(messages: List[Tuple[str, str]], window=12, task=None):
    new_messages = []
    for idx, (name, text) in tqdm(enumerate(messages), total=len(messages)):
        entry = {}
        max_idx = min(len(messages), idx + window)
        if task != None:
            entry['instruction'] = task
        else:
            entry['instruction'] = instruction
        entry['query'] = ""
        entry['output'] = ""
        text = ""
        for i in range(idx, max_idx):
            text += messages[i][0] + ": " + messages[i][1] + "\n"
        text = text[:-1]
        split_idx = np.random.randint(0, len(text) // 2, 1).item()
        entry['query'] = text[:split_idx]
        entry['output'] = text[split_idx:]
        new_messages.append(entry)
    return new_messages


def get_all_data(data_name: str):
    data = load(data_name)
    data = [(message['name'], message['text']) for message in data]
    data = convolute_messages(data)
    return data


def get_message_response_data(data_name: str):
    data = list(load(data_name))
    replies = filter(lambda x: 'reply' in x and 'name' in x, data)
    reply_map = {}
    for reply in replies:
        if reply['reply'] not in reply_map:
            reply_map[reply['reply']] = [reply]
        else:
            reply_map[reply['reply']].append(reply)
    trees = build_trees(data, reply_map)
    conversations = flatten_trees(trees)
    samples = []
    for conv in conversations:
        entry = {}
        entry['instruction'] = instruction
        entry['query'] = conv[0][0] + ": " + conv[0][1] + "\n"
        entry['output'] = ""
        for idx in range(1,len(conv)):
            entry['output'] +=conv[idx][0] + ": " + conv[idx][1] + "\n"
        entry['output'] = entry['output'][:-1]
        samples.append(entry)
    return samples


def build_trees(messages, reply_map):
    parents = {}
    childrens = []

    for message in messages:
        if message['message_id'] in reply_map:
            parent = Node(message['message_id'], n=message['name'], text=message['text'])
            if not parent.name in parents:
                parents[parent.name] = parent
            for child in reply_map[message['message_id']]:
                child_node = Node(child['message_id'], parent=parent, n=child['name'], text=child['text'])
                childrens.append(child_node)

    return parents


def flatten_trees(trees: List[Node]):
    def _flatten(parent: Node):
        if parent.is_leaf:
            return (parent.n, parent.text)
        else:
            return [(parent.n, parent.text)] + [_flatten(child) for child in parent.children]

    flattened_trees = []
    for parent in trees.values():
        flat_tree = _flatten(parent)
        flattened_trees.append(flat_tree)
    return flattened_trees
