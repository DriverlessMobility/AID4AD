import json
import os

f1 = open('../prior_map_train/prior_map.json')
f2 = open('../prior_map_val/prior_map.json')

content1 = f1.read()
content2 = f2.read()

a1 = json.loads(content1)
a2 = json.loads(content2)
print(len(a1), len(a2))

a3 = {}
for key, value in a1.items():
    a3[key] = os.path.split(value)[-1]
for key, value in a2.items():
    a3[key] = os.path.split(value)[-1]

print(len(a3))
b = json.dumps(a3)
f3 = open('map_prior.json', 'w')
f3.write(b)
f3.close()

