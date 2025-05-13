from pathlib import Path

header = """# Copyright 2025 [Chen Yi Low]
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
# limitations under the License.\n\n"""

for py_file in Path('.').rglob('*.py'):
    if 'insert_license.py' in str(py_file):  # Skip the script itself
        continue
    with open(py_file, 'r') as f:
        content = f.read()
    if 'Licensed under the Apache License' in content:
        continue  # Skip if already licensed
    with open(py_file, 'w') as f:
        f.write(header + content)
