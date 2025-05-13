# Copyright 2025 [Your Name]
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

import xml.etree.ElementTree as ET

tree = ET.parse('lang8_Eng_1.0.xml')
root = tree.getroot()

orig_sentences = []
correct_sentences = []
mother_tongues = []

for essay in root.findall('essay'):

    author = essay.find('author')
    if author is not None:
        mt_element = author.find('mother_tongue')
        mother_tongue = mt_element.text.strip() if mt_element is not None else "UNKNOWN"
    else:
        mother_tongue = "UNKNOWN"

    for sentence in essay.findall('sentence'):
        correct = sentence.find('correct')
        orig = sentence.find('orig')
        if correct is not None and orig is not None:
            correct_sentences.append(correct.text.strip())
            orig_sentences.append(orig.text.strip())
            mother_tongues.append(mother_tongue)

with open('orig.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(orig_sentences) + '\n')

with open('correct.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(correct_sentences) + '\n')

with open('mother_tongue.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(mother_tongues) + '\n')
