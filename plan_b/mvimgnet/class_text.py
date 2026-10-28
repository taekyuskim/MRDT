"""Class-name text conditioning: prompt normalization and precomputed T5-XXL embeddings.

MVImgNet2.0 has no captions; every object has a category label (mvimgnet2_category.json).
Each class name is normalized and turned into a prompt, encoded once with the T5-XXL encoder
RDT uses (google/t5-v1_1-xxl, d_model 4096), and stored in a small bank keyed by class_id.
Training/eval/sampling only read the bank, so they never need T5 itself.

    python -m plan_b.mvimgnet.class_text --catalog <catalog dir> --t5 <t5-v1_1-xxl dir> --out <bank.pt>
"""
import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import torch

TEMPLATE_VERSION = 'class_prompt_v2'
TYPO_FIXES = {'bbowknot': 'bowknot'}  # the only clear misspelling among the part-1 classes
# class_id -> name used instead of the dataset label (checked by viewing 80 random objects per class).
# 571 is labelled 'Origami frog' like 570 but holds fabric and paper flowers; 321/453 are both 'glue'
# (321 liquid glue bottles/tubes, some nail gel; 453 glue sticks and hot-melt sticks).
CLASS_NAME_OVERRIDES = {'571': 'artificial flower', '321': 'liquid glue', '453': 'glue stick'}
MAX_TEXT_TOKENS = 32


def normalize(name):
    """Lowercase, collapse whitespace, drop a leading 'the', apply the explicit typo table."""
    text = re.sub(r'\s+', ' ', str(name)).strip().lower()
    text = re.sub(r'^the ', '', text)
    return ' '.join(TYPO_FIXES.get(word, word) for word in text.split(' '))


def class_name(class_id, raw_name):
    return CLASS_NAME_OVERRIDES.get(str(class_id), raw_name)


def prompt(name):
    text = normalize(name)
    return f"a photo of {'an' if text[:1] in 'aeiou' else 'a'} {text}"


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


class TextBank:
    """class_id -> (MAX_TEXT_TOKENS x 4096 embedding, boolean mask); loaded lazily per process."""

    def __init__(self, path):
        self.path = str(path)
        self._data = None

    def _load(self):
        if self._data is None:
            self._data = torch.load(self.path, map_location='cpu', weights_only=True)
        return self._data

    def __getstate__(self):
        return {'path': self.path, '_data': None}

    @property
    def meta(self):
        return self._load()['meta']

    def lookup(self, class_id):
        data = self._load()
        i = data['index'][str(class_id)]
        return data['embeddings'][i], data['masks'][i]


def build(catalog_dir, t5_dir, out, device='cuda'):
    from transformers import AutoTokenizer, T5EncoderModel
    con = sqlite3.connect(f'file:{Path(catalog_dir) / "catalog.sqlite"}?mode=ro', uri=True)
    names = {}
    for (row_json,) in con.execute('SELECT row_json FROM instances'):
        row = json.loads(row_json)
        names.setdefault(str(row['class_id']), row['class_name'])
    class_ids = sorted(names, key=int)
    prompts = [prompt(class_name(c, names[c])) for c in class_ids]
    tokenizer = AutoTokenizer.from_pretrained(t5_dir)
    model = T5EncoderModel.from_pretrained(t5_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True).to(device).eval()
    tokens = tokenizer(prompts, max_length=MAX_TEXT_TOKENS, padding='max_length', truncation=True, return_tensors='pt')
    if (tokens.attention_mask.sum(1) >= MAX_TEXT_TOKENS).any():
        raise ValueError('A prompt reached MAX_TEXT_TOKENS; increase it')
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(prompts), 32):
            ids = tokens.input_ids[start:start + 32].to(device)
            mask = tokens.attention_mask[start:start + 32].to(device)
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
            embeddings.append((hidden * mask[..., None]).cpu())  # zero padded positions
    embeddings = torch.cat(embeddings)
    masks = tokens.attention_mask.bool()
    meta = {'template_version': TEMPLATE_VERSION, 'template': 'a photo of {a|an} {normalized class name}',
            'normalization': 'lowercase, collapse whitespace, strip leading "the", typo table',
            'typo_fixes': TYPO_FIXES, 'name_overrides': CLASS_NAME_OVERRIDES, 'max_tokens': MAX_TEXT_TOKENS, 'encoder': 'google/t5-v1_1-xxl T5EncoderModel fp32',
            't5_config_sha256': file_sha256(Path(t5_dir) / 'config.json'),
            'catalog_readiness_sha256': file_sha256(Path(catalog_dir) / 'readiness.json'),
            'classes': {c: {'raw': names[c], 'prompt': p, 'tokens': int(m.sum())}
                        for c, p, m in zip(class_ids, prompts, masks)}}
    torch.save({'meta': meta, 'index': {c: i for i, c in enumerate(class_ids)},
                'embeddings': embeddings.contiguous(), 'masks': masks}, out)
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--t5', type=Path, required=True, help='Local google/t5-v1_1-xxl directory')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    meta = build(args.catalog, args.t5, args.out, args.device)
    lengths = [v['tokens'] for v in meta['classes'].values()]
    print(json.dumps({'classes': len(meta['classes']), 'max_tokens_used': max(lengths), 'out': str(args.out),
                      'examples': [v['prompt'] for v in list(meta['classes'].values())[:5]]}))


if __name__ == '__main__':
    main()
