# Copyright 2025 Chen Yi Low
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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