import torch
import json
import re


acronym = {"oà": "òa", "oá": "óa", "oạ": "ọa", "oả": "ỏa", "oã": "õa",
           "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỵ": "ụy", "uỹ": "ũy",
           "aí": "ái", "oỉ": "ỏi"}

teen_text = {'ko': 'không', 'k': 'không', 'a': 'anh', 'e': 'em', 'bít': 'biết', 'h': 'giờ', 'j': 'gì',
             'mún': 'muốn', 'hok': 'học', 'iu': 'yêu', 'ck': 'ch', 'vk': 'vợ', 'ô': 'ông',
             'đc': 'được', 't': 'tôi', 'Ko': 'Không', 'A': 'Anh', 'E': 'Em', 'Bít': 'Biết',
             'H': 'Giờ', 'J': 'Gì', 'Mún': 'Muốn', 'Hok': 'Học', 'Iu': 'Yêu', 'Ck': 'Ch', 'Vk': 'Vợ', 'Ô': 'Ông',
             'Đc': 'Được', 'T': 'Tôi', 'f': 'ph', 'tk': 'th', 'nk': 'nh', 'F': 'Ph', 'Tk': 'Th', 'Nk': 'Nh',
             'zậy': 'vậy', 'mún': 'muốn'}  # more



def clean(text, vocab=None):
    cleanr = re.compile(
        r'<[^>]+>|<.*?>|&nbsp;|&amp;|&lt|p&gt|\u260e|<STYLE>(.*?)<\/STYLE>|<style>(.*?)<\/style>')
    text = re.sub(cleanr, ' ', text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub("[!\"#%&\\(\)\*\+,:;<=>?@[\]^_{|}~]]", ". ", text)
    text = re.sub("(\d+\/\d+\/\d+|\d+\/\d+)", " ", text)  # remove datetime
    inside_pattern = re.compile(r'\[.*\]|\(.*\)|\{.*\}')
    text = inside_pattern.sub(r'', text)
    text = text.replace(' - ', ' ')
    # text = " ".join(["".join([e for e in word if e in vocab ]) for word in text.split()])
    text = re.sub("\xa0", "", text)
    text = re.sub("\.+", '. ', text)
    text = re.sub("\s+", ' ', text)
    return text.strip()


# def convert2torch(*args):
#     res = tuple([torch.tensor(arg, dtype=torch.int64) for arg in args])
#     return res


if __name__ == "__main__":
    text = "dố là một lựa chọn k sai moà"
    new_text = []
    for t in text.split():
        if t in teen_text:
            new_text.append(teen_text[t])
        else:
            new_text.append(t)
    new_text = " ".join(new_text)
    text = clean(new_text)
    for arc in acronym.keys():
        if arc in text:
            text = text.replace(arc, acronym[arc])
    print(text)