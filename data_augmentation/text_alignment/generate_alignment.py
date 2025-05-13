import re
import json

count = 0
data = []
num_lines = sum(1 for line in open("sampleData/train/hansard.36.1.house.debates.001.e"))
for i in open("sample.out/int.alignOutput.giza"):
    count += 1
    if count % 3 == 0:
        tokens = re.findall(r'(\S+)\s+\(\{\s*([^}]*)\s*\}\)', i)
        alignments = []
        for target_idx, (target_word, source_positions_str) in enumerate(tokens, start=0):
            #print(target_idx, (target_word, source_positions_str))
            if source_positions_str.strip():  # Skip empty alignments
                source_positions = map(int, source_positions_str.strip().split())
                for source_idx in source_positions:
                    alignments.append([source_idx, target_idx])
        alignments = sorted(alignments, key=lambda x: x[0])
        data.append(alignments)
        #print(i, alignments, "\n")
    if int(count/3) == num_lines:
        break
with open("alignments.json", "w") as f:
    json.dump(data, f)