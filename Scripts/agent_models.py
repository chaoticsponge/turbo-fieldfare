"""Pinned specialist roles shared by the installer, router, and launcher."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESOURCES = ROOT / 'Sources/TurboFieldfareApp/Core/Resources'
ROLES = {
    'worker': {'catalog': RESOURCES / 'qwen-3.8-27b-8bit.json', 'directory': 'qwen3.8-27b-8bit.mlx', 'kind': 'vlm', 'temperature': 1.0, 'top_p': 0.95, 'top_k': 20},
    'coder': {'catalog': ROOT / 'Scripts/agent-models/coder.json', 'directory': 'Qwen3-Coder-30B-A3B-Instruct-4bit', 'kind': 'llm', 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20},
    'research': {'catalog': ROOT / 'Scripts/agent-models/research.json', 'directory': 'GLM-4.7-Flash-4bit', 'kind': 'llm', 'temperature': 0.7, 'top_p': 1.0, 'top_k': 0},
    'extract': {'catalog': RESOURCES / 'qwen-3.5-9b.json', 'directory': 'qwen3.5-9b-4bit.mlx', 'kind': 'vlm', 'temperature': 0.1, 'top_p': 0.95, 'top_k': 20},
    'embed': {'catalog': ROOT / 'Scripts/agent-models/embed.json', 'directory': 'Qwen3-Embedding-0.6B-8bit', 'kind': 'embedding'},
    'rerank': {'catalog': ROOT / 'Scripts/agent-models/rerank.json', 'directory': 'Qwen3-Reranker-0.6B-4bit', 'kind': 'reranker'},
}


def role_info(role, worker_precision='8bit'):
    info = dict(ROLES[role])
    if role == 'worker' and worker_precision == '3bit':
        info.update(catalog=RESOURCES / 'qwen-model.json', directory='qwen3.8-27b.mlx')
    return info


def catalog(role, worker_precision='8bit'):
    return json.loads(role_info(role, worker_precision)['catalog'].read_text())


def model_path(role, root=None, worker_precision='8bit'):
    return (root or ROOT / 'scratch') / role_info(role, worker_precision)['directory']


def route_config(roles, root=None, worker_precision='8bit'):
    return {role: {**{k: v for k, v in role_info(role, worker_precision).items() if k not in ('catalog', 'directory')},
                   'model': model_path(role, root, worker_precision).name,
                   'weight_bytes': sum(f['bytes'] for f in catalog(role, worker_precision)['files']
                                       if f['name'].endswith('.safetensors'))} for role in roles}


def weight_reservation(entry):
    return entry.get('resident_weight_bytes', entry.get('weight_bytes', 0)) * 1.05
