"""Native-Windows QLoRA policies, dynamic collation and bounded causal loss.

No torch/Transformers import is needed to inspect the corpus or configuration.
The loss bypasses CausalLM.forward: each backward chunk is recomputed separately,
so neither full vocabulary logits nor a retained graph per chunk is allocated.
"""
from functools import lru_cache
import json
from pathlib import Path

MAX_SEQUENCE_LENGTH = 8192
LOSS_CHUNK_TOKENS = 128
VRAM_LIMIT_BYTES = 14_500_000_000  # Decimal GB, deliberately not 14.5 GiB.
GRADIENT_ACCUMULATION = 8


def training_policy():
    return dict(per_device_train_batch_size=1, gradient_accumulation_steps=GRADIENT_ACCUMULATION,
                group_by_length=True, bf16=True, fp16=False, tf32=True,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                optim='paged_adamw_8bit', dataloader_num_workers=0, dataloader_pin_memory=True,
                dataloader_persistent_workers=False, remove_unused_columns=False)


class TokenizedTraces:
    """A map-style dataset; no multiprocess map or static 8192-token padding."""
    def __init__(self, rows, identities):
        self.rows = rows
        self.identities = identities
        self.lengths = [len(row['input_ids']) for row in rows]
        self.longest_index = max(range(len(rows)), key=self.lengths.__getitem__)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def tokenize_trajectories(path, tokenizer, max_length=MAX_SEQUENCE_LENGTH):
    rows, identities = [], []
    for line_number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        tokens = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=False)
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        if not isinstance(tokens, list) or any(not isinstance(v, int) for v in tokens):
            raise ValueError('Tokenizer must return one unpadded token list per trajectory')
        if len(tokens) > max_length:
            raise ValueError('Trajectory {} exceeds the declared {}-token context ({} tokens); refusing truncation'.format(
                line_number, max_length, len(tokens)))
        if len(tokens) < 2:
            raise ValueError('Trajectory {} has no causal prediction targets'.format(line_number))
        rows.append({'input_ids': tokens, 'attention_mask': [1] * len(tokens), 'labels': tokens.copy()})
        identities.append(str(row['source_event_id']))
    if not rows:
        raise ValueError('Empty tokenized instruction corpus')
    return TokenizedTraces(rows, identities)


class DynamicCausalCollator:
    def __init__(self, pad_token_id, max_length=MAX_SEQUENCE_LENGTH):
        self.pad_token_id = int(pad_token_id)
        self.max_length = max_length

    def __call__(self, features):
        import torch
        if not features:
            raise ValueError('Empty training batch')
        longest = max(len(row['input_ids']) for row in features)
        if longest > self.max_length:
            raise ValueError('Batch exceeds context; dynamic padding never truncates a trajectory')
        ids, masks, labels = [], [], []
        for row in features:
            length = len(row['input_ids'])
            target = row.get('labels', row['input_ids'])
            mask = row.get('attention_mask', [1] * length)
            if len(target) != length or len(mask) != length:
                raise ValueError('Training token, label and mask lengths differ')
            pad = longest - length
            ids.append(row['input_ids'] + [self.pad_token_id] * pad)
            masks.append(mask + [0] * pad)
            # Preserve real EOS tokens even when pad_token_id == eos_token_id.
            labels.append([t if m else -100 for t, m in zip(target, mask)] + [-100] * pad)
        return {'input_ids': torch.tensor(ids, dtype=torch.long),
                'attention_mask': torch.tensor(masks, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)}


@lru_cache(maxsize=1)
def _chunked_loss_function():
    import torch
    import torch.nn.functional as functional

    class ChunkedCausalCrossEntropy(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, weight, bias, labels, chunk_tokens):
            if weight.requires_grad or (bias is not None and bias.requires_grad):
                raise ValueError('Chunked frozen-head loss requires a frozen lm_head; target decoder LoRA modules only')
            if hidden.ndim != 3 or labels.shape != hidden.shape[:2] or weight.ndim != 2:
                raise ValueError('Expected hidden [batch, sequence, hidden], matching labels and a dense head')
            if hidden.shape[-1] != weight.shape[-1] or chunk_tokens <= 0:
                raise ValueError('Invalid head shape or loss chunk size')
            # Shift BEFORE chunking: targets at chunk boundaries must not vanish.
            targets = labels[:, 1:].reshape(-1)
            states = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])
            count = int((targets != -100).sum().item())
            if not count:
                raise ValueError('No unmasked causal targets')
            ctx.save_for_backward(hidden, weight, labels, bias if bias is not None else weight.new_empty(0))
            ctx.has_bias = bias is not None
            ctx.chunk_tokens, ctx.count = int(chunk_tokens), count
            total = torch.zeros((), device=hidden.device, dtype=torch.float32)
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                for start in range(0, len(targets), ctx.chunk_tokens):
                    end = start + ctx.chunk_tokens
                    logits = functional.linear(states[start:end].to(weight.dtype), weight, bias)
                    total.add_(functional.cross_entropy(logits.float(), targets[start:end], ignore_index=-100, reduction='sum'))
                    del logits
            return total / count

        @staticmethod
        def backward(ctx, grad_output):
            hidden, weight, labels, saved_bias = ctx.saved_tensors
            bias = saved_bias if ctx.has_bias else None
            targets = labels[:, 1:].reshape(-1)
            states = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])
            gradient = torch.zeros_like(states)
            # Only this one chunk's CE graph exists; autograd frees it immediately.
            for start in range(0, len(targets), ctx.chunk_tokens):
                end = start + ctx.chunk_tokens
                if not bool((targets[start:end] != -100).any().item()):
                    continue
                with torch.enable_grad(), torch.autocast(device_type=hidden.device.type, enabled=False):
                    chunk = states[start:end].detach().requires_grad_(True)
                    logits = functional.linear(chunk.to(weight.dtype), weight, bias)
                    loss = functional.cross_entropy(logits.float(), targets[start:end], ignore_index=-100, reduction='sum') / ctx.count
                    chunk_gradient = torch.autograd.grad(loss, chunk, grad_outputs=grad_output, create_graph=False)[0]
                gradient[start:end].copy_(chunk_gradient)
                del chunk, logits, loss, chunk_gradient
            result = torch.zeros_like(hidden)
            result[:, :-1, :] = gradient.reshape(hidden.shape[0], hidden.shape[1] - 1, hidden.shape[2])
            return result, None, None, None, None

    return ChunkedCausalCrossEntropy


def chunked_causal_loss(hidden, head, labels, chunk_tokens=LOSS_CHUNK_TOKENS):
    if getattr(head.weight, 'quant_state', None) is not None:
        raise ValueError('lm_head must remain dense BF16, not a packed quantized parameter')
    return _chunked_loss_function().apply(hidden, head.weight, getattr(head, 'bias', None), labels, chunk_tokens)


def backbone_loss(model, batch, chunk_tokens=LOSS_CHUNK_TOKENS):
    """Keep decoder LoRA layers active while bypassing full CausalLM logits/loss."""
    causal = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if getattr(causal.config, 'model_type', None) != 'qwen2':
        raise ValueError('The bounded-head training contract supports the Qwen2/Qwen2.5 architecture')
    import torch
    decoder = causal.get_decoder()
    device_type = batch['input_ids'].device.type
    # Calling the backbone directly bypasses Accelerate's CausalLM.forward AMP
    # wrapper, so explicitly retain BF16 activation compute on CUDA here.
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == 'cuda'):
        output = decoder(input_ids=batch['input_ids'], attention_mask=batch.get('attention_mask'),
                         use_cache=False, output_hidden_states=False, output_attentions=False, return_dict=True)
        return chunked_causal_loss(output.last_hidden_state, causal.get_output_embeddings(), batch['labels'], chunk_tokens)


def restore_frozen_embedding_dtypes(model, torch_module=None):
    if torch_module is None:
        import torch as torch_module
    # PEFT's k-bit preparation promotes nonquantized parameters to FP32. Restore
    # these two ~152k-vocabulary matrices; leave normalization parameters FP32.
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        if any(p.requires_grad for p in module.parameters()):
            raise ValueError('Embedding/lm_head must be frozen before restoring BF16')
        if getattr(module.weight, 'quant_state', None) is not None:
            raise ValueError('Expected dense input embedding and lm_head')
        module.to(dtype=torch_module.bfloat16)


def enable_fused_sdpa(torch_module):
    cuda = torch_module.backends.cuda
    previous = {'math': cuda.math_sdp_enabled(), 'flash': cuda.flash_sdp_enabled(),
                'efficient': cuda.mem_efficient_sdp_enabled()}
    cuda.enable_flash_sdp(True)
    cuda.enable_mem_efficient_sdp(True)
    cuda.enable_math_sdp(False)  # Never silently allocate quadratic math attention.
    if hasattr(cuda, 'cudnn_sdp_enabled'):
        previous['cudnn'] = cuda.cudnn_sdp_enabled()
        cuda.enable_cudnn_sdp(True)
    return previous


def restore_sdpa(torch_module, previous):
    cuda = torch_module.backends.cuda
    cuda.enable_math_sdp(previous['math'])
    cuda.enable_flash_sdp(previous['flash'])
    cuda.enable_mem_efficient_sdp(previous['efficient'])
    if 'cudnn' in previous:
        cuda.enable_cudnn_sdp(previous['cudnn'])
