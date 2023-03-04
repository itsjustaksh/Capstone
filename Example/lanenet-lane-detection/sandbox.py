import json

path = r"C:\Users\googl\Documents\Anything Relevant\University\4th Year\Project\pictureswithannotationfile\labels.json"
with open(path, 'r') as file:
        dict = json.load(file)

print(dict.keys())