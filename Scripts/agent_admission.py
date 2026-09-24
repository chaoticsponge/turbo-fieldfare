"""Model-free session estimates; runtime memory guards remain authoritative."""
from dataclasses import dataclass
import json
import math

GIB = 1024**3


def cache_profile(config):
    """Match the pinned MLX GQA, Qwen hybrid and GLM latent cache layouts.

    No quantized-KV saving is assumed. Unknown layouts retain legacy headroom.
    """
    c = config.get('text_config', config)
    kind = c.get('model_type')
    try:
        def positive(key):
            value = c[key]
            if type(value) is not int or value <= 0:
                raise ValueError(key)
            return value
        layers, hidden = positive('num_hidden_layers'), positive('hidden_size')
        dtype = c.get('dtype', c.get('torch_dtype', config.get('dtype', config.get('torch_dtype'))))
        width = 2 if dtype in ('float16', 'bfloat16') else 4
        state = 0
        attention_workspace = 0
        if kind == 'glm4_moe_lite':
            per_token = layers * (positive('kv_lora_rank') + positive('qk_rope_head_dim')) * width
            layout = 'mla'
            # GLM explicitly materializes positional attention scores.
            attention_workspace = 2 * width * positive('num_attention_heads')
        elif kind in ('qwen3', 'qwen3_moe', 'qwen3_5_text', 'qwen3_5'):
            attention_layers = layers
            layout = 'gqa'
            if kind in ('qwen3_5_text', 'qwen3_5'):
                attention_layers = layers // positive('full_attention_interval')
                linear_layers = layers - attention_layers
                vh, kh = positive('linear_num_value_heads'), positive('linear_num_key_heads')
                kd, vd = positive('linear_key_head_dim'), positive('linear_value_head_dim')
                # Delta state is FP32; convolution state follows activation dtype.
                state = linear_layers * (vh * kd * vd * 4 +
                    (positive('linear_conv_kernel_dim') - 1) * (2 * kh * kd + vh * vd) * width)
                layout = 'hybrid'
            dim = c.get('head_dim') or hidden // positive('num_attention_heads')
            if type(dim) is not int or dim <= 0:
                raise ValueError('head_dim')
            per_token = attention_layers * 2 * positive('num_key_value_heads') * dim * width
        else:
            return {'layout': 'fallback'}
        return {'layout': layout, 'kv_bytes_per_token': per_token,
                'state_bytes': state, 'hidden_size': hidden,
                'attention_workspace_bytes_per_pair': attention_workspace}
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return {'layout': 'fallback'}


@dataclass(frozen=True)
class SessionEstimate:
    session_bytes: int
    total_tokens: int | None
    prompt_tokens: int | None
    method: str


def estimate_session(path, body, route):
    profile = route.get('cache_profile', {})
    limit = route.get('max_context_window')
    if not limit or profile.get('layout', 'fallback') == 'fallback':
        return SessionEstimate(3 * GIB, None, None, 'fixed-fallback')
    # Retrieval has no autoregressive KV lineage. Retain existing conservative
    # workspace admission until its batched activation memory is measured.
    if path != '/v1/chat/completions':
        return SessionEstimate(3 * GIB, None, None, 'retrieval-workspace')
    output = body['max_tokens']  # normalized and validated by resolve()
    messages = body['messages']
    images = any(isinstance(m, dict) and isinstance(m.get('content'), list)
                 and any(isinstance(p, dict) and p.get('type') in ('image_url', 'input_image')
                         for p in m['content']) for m in messages)
    if images:
        prompt = limit - output
        method = 'image-context-ceiling'
    else:
        # UTF-8 bytes deliberately overestimate most text token counts. Include
        # complete history, tool schemas and template options, not just user text.
        # Extra template tokens are bounded again by the actual engine validator.
        payload = {k: body[k] for k in ('messages', 'tools', 'tool_choice',
                   'response_format', 'chat_template_kwargs') if k in body}
        encoded = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        prompt = min(limit - output, len(encoded) + 1024 + 128 * len(messages))
        method = 'utf8-upper-estimate'
    total = prompt + output
    rounded = math.ceil(total / 256) * 256
    cache = rounded * profile['kv_bytes_per_token'] + profile['state_bytes']
    # Per-request workspace floor plus bounded prefill allowance; these are
    # planning margins, not measured maxima for every kernel/model combination.
    prefill = min(prompt, 2048) * (profile['hidden_size'] * 32 +
        prompt * profile.get('attention_workspace_bytes_per_pair', 0))
    size = GIB + math.ceil(cache * 1.25) + prefill + (GIB if images else 0)
    return SessionEstimate(size, total, prompt, method)
