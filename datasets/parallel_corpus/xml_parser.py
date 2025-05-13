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
