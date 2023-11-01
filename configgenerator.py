import json

jsondata = {
    "word2vecfile": "/Users/berlin/CUHK/text-segmentation/data/word2vec/GoogleNews-vectors-negative300.bin",
    "choidataset": "/home/omri/code/text-segmentation-2017/data/choi",
    "wikidataset": "/Users/berlin/CUHK/text-segmentation/data/wiki2"
}

with open('config.json', 'w') as f:
    json.dump(jsondata, f)
