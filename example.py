from pathlib import Path

for py_file in Path('.').rglob('*.py'):
    if 'insert_license.py' in str(py_file):  # Skip the script itself
        continue
    with open(py_file, 'r') as f:
        content = f.read()
    if 'Chen Yi Low' in content:
        content = content.replace('Chen Yi Low', 'Chen Yi Low')
        with open(py_file, 'w') as f:
            f.write(content)
