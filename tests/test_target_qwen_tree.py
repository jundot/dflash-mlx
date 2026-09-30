# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models import qwen3, qwen3_5, qwen3_next

from dflash_mlx.engine.ddtree import (
    build_flat_ddtree,
    build_flat_tree_inputs,
    flat_tree_path_token_ids,
)
from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps


def _tiny_qwen3_model():
    args = qwen3.ModelArgs(
        model_type="qwen3",
        hidden_size=16,
        num_hidden_layers=2,
        intermediate_size=32,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=1,
        max_position_embeddings=128,
        rope_theta=10000.0,
        head_dim=8,
        tie_word_embeddings=True,
    )
    return qwen3.Model(args)


def _tiny_qwen35_hybrid_model():
    args = qwen3_5.ModelArgs(
        model_type="qwen3_5",
        text_config={
            "model_type": "qwen3_5",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 64,
            "tie_word_embeddings": True,
            "max_position_embeddings": 128,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 2,
            "rope_parameters": {
                "type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
        },
    )
    return qwen3_5.Model(args)


def _tiny_qwen3_next_model():
    args = qwen3_next.ModelArgs(
        model_type="qwen3_next",
        hidden_size=16,
        num_hidden_layers=2,
        intermediate_size=32,
        num_attention_heads=2,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=32,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        num_experts=0,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        shared_expert_intermediate_size=32,
        mlp_only_layers=[],
        moe_intermediate_size=32,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=1,
        rope_theta=10000.0,
        partial_rotary_factor=1.0,
        max_position_embeddings=128,
        head_dim=8,
        tie_word_embeddings=True,
        full_attention_interval=2,
    )
    return qwen3_next.Model(args)


def _prefilled_cache(model, ops: QwenGdnTargetOps, prompt: mx.array):
    cache = ops.make_cache(
        model,
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
        target_fa_window=0,
    )
    ops.forward_with_hidden_capture(model, input_ids=prompt, cache=cache)
    return cache


def _assert_close(lhs: mx.array, rhs: mx.array, *, atol: float = 1e-4) -> None:
    mx.eval(lhs, rhs)
    assert float(mx.max(mx.abs(lhs - rhs)).item()) <= atol


def test_qwen_tree_verify_matches_sequential_path_logits():
    model = _tiny_qwen3_model()
    ops = QwenGdnTargetOps()
    prompt = mx.array([[1, 2, 3]], dtype=mx.uint32)
    tree = build_flat_ddtree(
        top_token_ids_desc=[
            [5, 6],
            [7, 8],
        ],
        top_scores_desc=[
            [-0.1, -1.0],
            [-0.2, -0.3],
        ],
        budget=3,
    )
    tree_inputs = build_flat_tree_inputs(tree, root_token_id=4, prefix_len=3)

    tree_cache = _prefilled_cache(model, ops, prompt)
    tree_logits, tree_captured = ops.verify_tree_block(
        target_model=model,
        tree_inputs=tree_inputs,
        target_cache=tree_cache,
        capture_layer_ids={1, 2},
    )
    mx.eval(tree_logits, *tree_captured.values())

    for slot_index, path in enumerate(flat_tree_path_token_ids(tree, root_token_id=4)):
        path_cache = _prefilled_cache(model, ops, prompt)
        path_logits, _ = ops.verify_block(
            target_model=model,
            verify_ids=mx.array([path], dtype=mx.uint32),
            target_cache=path_cache,
            capture_layer_ids={1, 2},
        )
        _assert_close(tree_logits[:, slot_index, :], path_logits[:, -1, :], atol=5e-3)


def test_qwen_tree_commit_gathers_accepted_sibling_path_cache():
    model = _tiny_qwen3_model()
    ops = QwenGdnTargetOps()
    prompt = mx.array([[1, 2, 3]], dtype=mx.uint32)
    tree = build_flat_ddtree(
        top_token_ids_desc=[
            [5, 6],
            [7, 8],
        ],
        top_scores_desc=[
            [-0.1, -1.0],
            [-0.2, -0.3],
        ],
        budget=3,
    )
    tree_inputs = build_flat_tree_inputs(tree, root_token_id=4, prefix_len=3)

    tree_cache = _prefilled_cache(model, ops, prompt)
    ops.verify_tree_block(
        target_model=model,
        tree_inputs=tree_inputs,
        target_cache=tree_cache,
        capture_layer_ids={1},
    )
    ops.restore_after_tree_acceptance(
        tree_cache,
        accepted_tree_indices=[0, 1, 3],
    )

    sequential_cache = _prefilled_cache(model, ops, prompt)
    ops.verify_block(
        target_model=model,
        verify_ids=mx.array([[4, 5, 8]], dtype=mx.uint32),
        target_cache=sequential_cache,
        capture_layer_ids={1},
    )

    next_id = mx.array([[9]], dtype=mx.uint32)
    tree_next, _ = ops.verify_block(
        target_model=model,
        verify_ids=next_id,
        target_cache=tree_cache,
        capture_layer_ids={1},
    )
    sequential_next, _ = ops.verify_block(
        target_model=model,
        verify_ids=next_id,
        target_cache=sequential_cache,
        capture_layer_ids={1},
    )

    _assert_close(tree_next, sequential_next)


def test_qwen_hybrid_tree_verify_matches_sequential_path_logits():
    model = _tiny_qwen35_hybrid_model()
    ops = QwenGdnTargetOps()
    prompt = mx.array([[1, 2, 3]], dtype=mx.uint32)
    tree = build_flat_ddtree(
        top_token_ids_desc=[
            [5, 6],
            [7, 8],
        ],
        top_scores_desc=[
            [-0.1, -1.0],
            [-0.2, -0.3],
        ],
        budget=3,
    )
    tree_inputs = build_flat_tree_inputs(tree, root_token_id=4, prefix_len=3)

    tree_cache = _prefilled_cache(model, ops, prompt)
    tree_logits, _tree_captured = ops.verify_tree_block(
        target_model=model,
        tree_inputs=tree_inputs,
        target_cache=tree_cache,
        capture_layer_ids={1, 2},
    )
    mx.eval(tree_logits)

    for slot_index, path in enumerate(flat_tree_path_token_ids(tree, root_token_id=4)):
        path_cache = _prefilled_cache(model, ops, prompt)
        path_logits, _ = ops.verify_block(
            target_model=model,
            verify_ids=mx.array([path], dtype=mx.uint32),
            target_cache=path_cache,
            capture_layer_ids={1, 2},
        )
        tree_slot_logits = tree_logits[:, slot_index, :]
        path_last_logits = path_logits[:, -1, :]
        _assert_close(tree_slot_logits, path_last_logits, atol=1e-2)
        assert int(mx.argmax(tree_slot_logits, axis=-1).item()) == int(
            mx.argmax(path_last_logits, axis=-1).item()
        )


def test_qwen_hybrid_tree_verify_handles_empty_recurrent_prefix():
    model = _tiny_qwen35_hybrid_model()
    ops = QwenGdnTargetOps()
    tree = build_flat_ddtree(
        top_token_ids_desc=[
            [5, 6],
            [7, 8],
        ],
        top_scores_desc=[
            [-0.1, -1.0],
            [-0.2, -0.3],
        ],
        budget=3,
    )
    tree_inputs = build_flat_tree_inputs(tree, root_token_id=4, prefix_len=0)

    tree_cache = ops.make_cache(
        model,
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
        target_fa_window=0,
    )
    tree_logits, _tree_captured = ops.verify_tree_block(
        target_model=model,
        tree_inputs=tree_inputs,
        target_cache=tree_cache,
        capture_layer_ids={1, 2},
    )
    mx.eval(tree_logits)

    for slot_index, path in enumerate(flat_tree_path_token_ids(tree, root_token_id=4)):
        path_cache = ops.make_cache(
            model,
            enable_speculative_linear_cache=True,
            quantize_kv_cache=False,
            target_fa_window=0,
        )
        path_logits, _ = ops.verify_block(
            target_model=model,
            verify_ids=mx.array([path], dtype=mx.uint32),
            target_cache=path_cache,
            capture_layer_ids={1, 2},
        )
        _assert_close(tree_logits[:, slot_index, :], path_logits[:, -1, :], atol=1e-2)


def test_qwen3_next_rollback_verify_uses_fused_gdn_projection():
    model = _tiny_qwen3_next_model()
    ops = QwenGdnTargetOps()
    prompt = mx.array([[1, 2, 3]], dtype=mx.uint32)
    verify_ids = mx.array([[4, 5]], dtype=mx.uint32)
    next_id = mx.array([[6]], dtype=mx.uint32)

    rollback_cache = _prefilled_cache(model, ops, prompt)
    comparison_cache = _prefilled_cache(model, ops, prompt)
    sequential_cache = _prefilled_cache(model, ops, prompt)
    ops.arm_rollback(rollback_cache, prefix_len=int(prompt.shape[1]))

    rollback_logits, _ = ops.verify_block(
        target_model=model,
        verify_ids=verify_ids,
        target_cache=rollback_cache,
        capture_layer_ids={1, 2},
    )
    sequential_logits, _ = ops.verify_block(
        target_model=model,
        verify_ids=verify_ids,
        target_cache=comparison_cache,
        capture_layer_ids={1, 2},
    )

    _assert_close(rollback_logits, sequential_logits, atol=1e-2)
    assert any(getattr(cache_entry, "_tape", None) is not None for cache_entry in rollback_cache)

    ops.restore_after_acceptance(
        rollback_cache,
        target_len=int(prompt.shape[1]) + 1,
        acceptance_length=0,
        drafted_tokens=1,
    )
    ops.verify_block(
        target_model=model,
        verify_ids=verify_ids[:, :1],
        target_cache=sequential_cache,
        capture_layer_ids={1, 2},
    )
    rollback_next, _ = ops.verify_block(
        target_model=model,
        verify_ids=next_id,
        target_cache=rollback_cache,
        capture_layer_ids={1, 2},
    )
    sequential_next, _ = ops.verify_block(
        target_model=model,
        verify_ids=next_id,
        target_cache=sequential_cache,
        capture_layer_ids={1, 2},
    )
    _assert_close(rollback_next, sequential_next, atol=1e-2)


def test_qwen3_next_tree_verify_and_commit_use_fused_gdn_projection():
    model = _tiny_qwen3_next_model()
    ops = QwenGdnTargetOps()
    prompt = mx.array([[1, 2, 3]], dtype=mx.uint32)
    tree = build_flat_ddtree(
        top_token_ids_desc=[
            [5, 6],
            [7, 8],
        ],
        top_scores_desc=[
            [-0.1, -1.0],
            [-0.2, -0.3],
        ],
        budget=3,
    )
    tree_inputs = build_flat_tree_inputs(tree, root_token_id=4, prefix_len=3)

    tree_cache = _prefilled_cache(model, ops, prompt)
    tree_logits, _tree_captured = ops.verify_tree_block(
        target_model=model,
        tree_inputs=tree_inputs,
        target_cache=tree_cache,
        capture_layer_ids={1, 2},
    )
    mx.eval(tree_logits)

    for slot_index, path in enumerate(flat_tree_path_token_ids(tree, root_token_id=4)):
        path_cache = _prefilled_cache(model, ops, prompt)
        path_logits, _ = ops.verify_block(
            target_model=model,
            verify_ids=mx.array([path], dtype=mx.uint32),
            target_cache=path_cache,
            capture_layer_ids={1, 2},
        )
        _assert_close(tree_logits[:, slot_index, :], path_logits[:, -1, :], atol=1e-2)

    ops.restore_after_tree_acceptance(
        tree_cache,
        accepted_tree_indices=[0, 1, 3],
    )
    sequential_cache = _prefilled_cache(model, ops, prompt)
    ops.verify_block(
        target_model=model,
        verify_ids=mx.array([[4, 5, 8]], dtype=mx.uint32),
        target_cache=sequential_cache,
        capture_layer_ids={1, 2},
    )
    next_id = mx.array([[9]], dtype=mx.uint32)
    tree_next, _ = ops.verify_block(
        target_model=model,
        verify_ids=next_id,
        target_cache=tree_cache,
        capture_layer_ids={1, 2},
    )
    sequential_next, _ = ops.verify_block(
        target_model=model,
        verify_ids=next_id,
        target_cache=sequential_cache,
        capture_layer_ids={1, 2},
    )
    _assert_close(tree_next, sequential_next, atol=1e-2)


def test_qwen_hybrid_tree_commit_gathers_accepted_sibling_path_cache():
    model = _tiny_qwen35_hybrid_model()
    ops = QwenGdnTargetOps()
    prompt = mx.array([[1, 2, 3]], dtype=mx.uint32)
    tree = build_flat_ddtree(
        top_token_ids_desc=[
            [5, 6],
            [7, 8],
        ],
        top_scores_desc=[
            [-0.1, -1.0],
            [-0.2, -0.3],
        ],
        budget=3,
    )
    tree_inputs = build_flat_tree_inputs(tree, root_token_id=4, prefix_len=3)

    tree_cache = _prefilled_cache(model, ops, prompt)
    ops.verify_tree_block(
        target_model=model,
        tree_inputs=tree_inputs,
        target_cache=tree_cache,
        capture_layer_ids={1},
    )
    ops.restore_after_tree_acceptance(
        tree_cache,
        accepted_tree_indices=[0, 1, 3],
    )

    sequential_cache = _prefilled_cache(model, ops, prompt)
    ops.verify_block(
        target_model=model,
        verify_ids=mx.array([[4, 5, 8]], dtype=mx.uint32),
        target_cache=sequential_cache,
        capture_layer_ids={1},
    )

    next_id = mx.array([[9]], dtype=mx.uint32)
    tree_next, _ = ops.verify_block(
        target_model=model,
        verify_ids=next_id,
        target_cache=tree_cache,
        capture_layer_ids={1},
    )
    sequential_next, _ = ops.verify_block(
        target_model=model,
        verify_ids=next_id,
        target_cache=sequential_cache,
        capture_layer_ids={1},
    )

    _assert_close(tree_next, sequential_next, atol=7e-3)


def test_qwen_hybrid_armed_verify_normalizes_qk_like_stock():
    model = _tiny_qwen35_hybrid_model()
    # Tiny k rows make the l2norm eps visible in the logits.
    for layer in model.language_model.model.layers:
        if not getattr(layer, "is_linear", False):
            continue
        gdn = layer.linear_attn
        rows = mx.arange(gdn.in_proj_qkv.weight.shape[0])
        k_rows = (rows >= gdn.key_dim) & (rows < 2 * gdn.key_dim)
        gdn.in_proj_qkv.weight = gdn.in_proj_qkv.weight * mx.where(
            k_rows, 1e-3, 1.0
        )[:, None]
    ops = QwenGdnTargetOps()
    prompt = mx.array([[1, 2, 3]], dtype=mx.uint32)
    verify_ids = mx.array([[4, 5]], dtype=mx.uint32)

    armed_cache = _prefilled_cache(model, ops, prompt)
    stock_cache = _prefilled_cache(model, ops, prompt)
    ops.arm_rollback(armed_cache, prefix_len=int(prompt.shape[1]))

    armed_logits, _ = ops.verify_block(
        target_model=model,
        verify_ids=verify_ids,
        target_cache=armed_cache,
        capture_layer_ids={1, 2},
    )
    stock_logits, _ = ops.verify_block(
        target_model=model,
        verify_ids=verify_ids,
        target_cache=stock_cache,
        capture_layer_ids={1, 2},
    )
    _assert_close(armed_logits, stock_logits)
