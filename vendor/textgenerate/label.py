import os
import pprint
import json
import collections
import operator

folder = "out"

def path_to_files(dir_folder):
    list_of_audio = []
    content_dict = {}
    for file in os.listdir(dir_folder):
        if file.endswith(".jpg"):
            #path_directory = "%s/%s" % (dir_folder, file)
            #print (file)
            file_num = int(file.partition("_")[0])
            file_content = file.partition("_")[2].strip(".jpg")
            file_len = len(file_content)
            content_dict.update({file_num:{'label':file_content,'length':file_len}})
            #list_of_audio.append(path_directory)
    return content_dict

content = path_to_files(folder)
sort_content = collections.OrderedDict(sorted(content.items()))
pprint.pprint (sort_content)
with open('label.json','w') as outfile:
    json.dump(sort_content,outfile)

