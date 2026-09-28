"""Build notebook cells from staged markdown while protecting literal code."""
from pathlib import Path
import json
import re
from urllib.parse import quote

base = Path(__file__).resolve().parents[2]
source = base / '6. Numerical Computing with NumPy and PyTorch.md'
destination = base / 'Sources/Notebooks/NumPy and PyTorch examples.ipynb'
original = json.loads(destination.read_text())
text = re.sub(r'\A---\n.*?\n---\n\n', '', source.read_text(), count=1, flags=re.S)
# Obsidian displays the filename; a standalone notebook needs its own title.
if not re.match(r'^# ', text):
    title = re.sub(r'^\d+\.\s*', '', source.stem)
    text = '# ' + title + '\n\n' + text.lstrip()

# Obsidian block IDs placed after appendix callouts name those appendices.
# Links to them become links to the notebook headings made from the callouts.
blocks = {}
title = None
for line in text.splitlines():
    callout = re.match(r'^> \[!note\]- (.*)$', line)
    if callout:
        title = callout.group(1)
    block = re.match(r'^\^([\w-]+)\s*$', line)
    if block and title:
        blocks[block.group(1)] = title

def prose_links(text):
    # Restrict wiki-link processing to prose, never inline code.
    segments = re.split(r'(`+[^`]*`+)', text)
    def image(match):
        # Figures live in Sources/Images, next to this notebook's folder.
        name, width = match.group(1), match.group(2) or '680'
        return f'<img src="../Images/{quote(name)}" width="{width}" alt="">'
    def link(match):
        target, label = match.group(1), match.group(2)
        if target.startswith('#^') and target[2:] in blocks:
            return f'[{label}](#' + blocks[target[2:]].replace(' ', '-') + ')'
        if target.startswith('ML Mastery Notes/0. Foundations/'):
            target = target.removeprefix('ML Mastery Notes/0. Foundations/')
        path, marker, anchor = target.partition('#')
        if not path.lower().endswith(('.md', '.pdf', '.ipynb')):
            path += '.md'
        href = '../../' + quote(path, safe='/')
        if marker:
            href += '#' + quote(anchor.lower().replace(' ', '-'))
        return f'[{label}]({href})'
    for i in range(0, len(segments), 2):
        segments[i] = re.sub(r'!\[\[ML Mastery Notes/0\. Foundations/Sources/Images/([^\]|]+)(?:\|(\d+))?\]\]',
                             image, segments[i])
        segments[i] = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', link, segments[i])
        segments[i] = segments[i].replace('(Sources/Exercises/', '(../Exercises/')
        # Jupyter's local heading IDs use hyphens; Obsidian accepts spaces.
        segments[i] = re.sub(r'\]\(#([^)]*)\)',
                             lambda m: '](#' + m.group(1).replace('%20', '-') + ')',
                             segments[i])
    return ''.join(segments)

# Decode Obsidian blockquotes before splitting fenced code. Answers remain
# collapsible; appendix headings become ordinary notebook headings.
normalized = []
in_callout = False
in_answers = False
for line in text.splitlines():
    callout = re.match(r'^> \[!(note|answer)\]- (.*)$', line)
    if callout:
        in_callout = True
        in_answers = callout.group(1) == 'answer'
        if in_answers:
            normalized.extend(['<details>', '<summary>Check your answers</summary>', ''])
        else:
            normalized.extend(['### ' + callout.group(2), ''])
    elif in_callout and line.startswith('>'):
        normalized.append(line[2:] if line.startswith('> ') else line[1:])
    elif re.match(r'^\^[\w-]+\s*$', line):
        # Block IDs are Obsidian link targets, not notebook text.
        in_callout = False
    else:
        if in_answers:
            normalized.extend(['', '</details>', ''])
        in_callout = False
        in_answers = False
        normalized.append(line)
if in_answers:
    normalized.extend(['', '</details>'])
text = '\n'.join(normalized) + '\n'

def cell(kind, body):
    result = {'cell_type': kind, 'metadata': {}, 'source': body.splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result

intro, remainder = text.split('## Arrays and numerical computation', 1)
startup = '''import platform
from importlib.metadata import version
import numpy as np
import torch

# Keep this small numerical lab on a reproducible CPU execution path.
torch.set_num_threads(1)
print(f"Python {platform.python_version()} | {platform.system()} {platform.machine()} | CPU")
for package in ("numpy", "scipy", "matplotlib", "torch", "jupyterlab",
                "ipykernel", "nbclient", "nbformat"):
    print(f"{package}: {version(package)}")

array = np.array([1.0, 2.0], dtype=np.float64)
shared = torch.from_numpy(array)
array[0] = 3.0
assert shared.device.type == "cpu" and shared.dtype == torch.float64
assert shared[0].item() == 3.0
print(f"NumPy/PyTorch shared storage: OK | CPU threads: {torch.get_num_threads()}")
'''
cells = [cell('markdown', prose_links(intro.strip() + '\n\nRun this environment check, then run the remaining cells in order.\n')),
         cell('code', startup)]
parts = re.split(r'^```python\n(.*?)^```\s*$', '## Arrays and numerical computation' + remainder,
                 flags=re.S | re.M)
for i, part in enumerate(parts):
    if part.strip():
        cells.append(cell('code' if i % 2 else 'markdown',
                          part.strip() + '\n' if i % 2 else prose_links(part.strip() + '\n')))

old_code = [''.join(c['source']).strip() for c in original['cells'] if c['cell_type'] == 'code']
if old_code and old_code[0].startswith('import platform\n'):
    old_code = old_code[1:]
new_code = [''.join(c['source']).strip() for c in cells if c['cell_type'] == 'code'][1:]
# Changed code receives no stale execution output. Unchanged code keeps its results.
prior_code = {''.join(c['source']).strip(): c for c in original['cells'] if c['cell_type'] == 'code'}
preserved_outputs = 0
for content in cells:
    if content['cell_type'] == 'code':
        prior = prior_code.get(''.join(content['source']).strip())
        if prior is not None:
            # Preserve the entire unchanged code cell, including its exact
            # source representation, outputs, execution metadata, and ID.
            content.clear()
            content.update(prior)
            preserved_outputs += 1
assert 'X[[0, 2]] = 0' in ''.join(''.join(c['source']) for c in cells if c['cell_type'] == 'markdown')
metadata = original['metadata']
metadata['kernelspec'] = {'display_name': 'Python (Foundations CPU)', 'language': 'python', 'name': 'foundations'}
for i, content in enumerate(cells):
    content.setdefault('id', f'foundations-{i:02d}')
notebook = {'cells': cells, 'metadata': metadata, 'nbformat': 4, 'nbformat_minor': 5}
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
print(f'Synchronized {len(cells)} cells; retained outputs for {preserved_outputs} unchanged code cells.')
