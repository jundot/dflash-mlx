from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from dflash_mlx.engine.acceptance import rejection_sample
from dflash_mlx.engine.sampling import sampling_probs
from dflash_mlx.model import (
    CandidateSelector,
    DFlash2DraftModel,
    DFlashAttention,
    DFlashDraftModelArgs,
    _grouped_dynamic_convolve,
)
from dflash_mlx.runtime.loading import _get_dflash_model_classes, load_draft_bundle


def _args(**overrides):
    values = dict(
        model_type="qwen3",
        hidden_size=4,
        num_hidden_layers=1,
        intermediate_size=8,
        num_attention_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=4,
        num_key_value_heads=1,
        max_position_embeddings=128,
        rope_theta=10_000.0,
        head_dim=2,
        tie_word_embeddings=False,
        num_target_layers=8,
        block_size=4,
        layer_types=("sliding_attention",),
        sliding_window=4,
        is_causal=False,
        dflash_config={
            "target_layer_ids": [1],
            "mask_token_id": 3,
            "conv_kernel_size": 2,
            "conv_group_size": 2,
            "selector_rank": 1,
            "selector_top_k": 2,
        },
    )
    values.update(overrides)
    return DFlashDraftModelArgs(**values)


def test_dflash2_config_normalizes_nested_fields():
    config = {
        key: value
        for key, value in _args().__dict__.items()
        if key not in ("block_size", "rope_theta", "rope_scaling")
    }
    config.update(
        architectures=["DFlash2DraftModel"],
        rope_parameters={"rope_theta": 500_000.0, "rope_type": "default"},
    )
    config["dflash_config"] = {**config["dflash_config"], "block_size": 16}
    args = DFlashDraftModelArgs.from_dict(config)
    assert args.block_size == 16
    assert args.rope_theta == 500_000.0
    assert args.rope_scaling is None
    assert args.is_causal is False


def test_dflash2_config_preserves_root_values():
    config = dict(_args().__dict__)
    config["rope_parameters"] = {
        "rope_theta": 500_000.0,
        "rope_type": "default",
    }
    config["dflash_config"] = {**config["dflash_config"], "block_size": 16}
    args = DFlashDraftModelArgs.from_dict(config)
    assert args.block_size == 4
    assert args.rope_theta == 10_000.0


def test_dflash2_config_normalizes_scaled_rope():
    config = {
        key: value
        for key, value in _args().__dict__.items()
        if key not in ("rope_theta", "rope_scaling")
    }
    config["rope_parameters"] = {
        "rope_theta": 500_000.0,
        "rope_type": "yarn",
        "factor": 4.0,
    }
    args = DFlashDraftModelArgs.from_dict(config)
    assert args.rope_theta == 500_000.0
    assert args.rope_scaling == {"rope_type": "yarn", "factor": 4.0}


def test_dflash2_config_does_not_invent_required_values():
    config = {
        key: value
        for key, value in _args().__dict__.items()
        if key not in ("block_size", "rope_theta", "rope_scaling")
    }
    with pytest.raises(TypeError):
        DFlashDraftModelArgs.from_dict(config)


def test_dflash2_loader_dispatches_by_architecture():
    model_cls, args_cls = _get_dflash_model_classes(
        {"model_type": "qwen3", "architectures": ["DFlash2DraftModel"]}
    )
    assert model_cls is DFlash2DraftModel
    assert args_cls is DFlashDraftModelArgs


def test_dflash2_loader_accepts_bare_codebook_keys(tmp_path):
    args = _args()
    weights = dict(tree_flatten(DFlash2DraftModel(args).parameters()))
    for name in ("predecessor_codebook", "successor_codebook"):
        key = f"candidate_selector.{name}"
        weights[key] = weights.pop(f"{key}.weight")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(
        json.dumps({**args.__dict__, "architectures": ["DFlash2DraftModel"]})
    )

    model, _ = load_draft_bundle(tmp_path, lazy=False)

    assert model.candidate_selector.predecessor_codebook.weight.shape == (4, 1)
    assert model.candidate_selector.successor_codebook.weight.shape == (4, 1)


def test_noncausal_sliding_mask_sees_whole_block_and_windowed_context():
    attention = DFlashAttention(_args(), 0)
    mask = attention._attention_mask(
        block_len=4,
        query_offset=10,
        key_len=7,
        key_positions=mx.array([7, 8, 9, 10, 11, 12, 13]),
    )
    assert mask.tolist() == [
        [True, True, True, True, True, True, True],
        [False, True, True, True, True, True, True],
        [False, False, True, True, True, True, True],
        [False, False, False, True, True, True, True],
    ]


def test_causal_sliding_mask_preserves_legacy_dflash_behavior():
    attention = DFlashAttention(_args(is_causal=None), 0)
    mask = attention._attention_mask(
        block_len=4,
        query_offset=10,
        key_len=7,
        key_positions=mx.array([7, 8, 9, 10, 11, 12, 13]),
    )
    assert mask.tolist() == [
        [True, True, True, True, False, False, False],
        [False, True, True, True, True, False, False],
        [False, False, True, True, True, True, False],
        [False, False, False, True, True, True, True],
    ]


def test_grouped_dynamic_convolution_is_block_local_and_causal():
    hidden = mx.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    base = mx.array([[1.0, 1.0], [10.0, 10.0]])
    dynamic = mx.zeros((1, 3, 2, 1))
    output = _grouped_dynamic_convolve(hidden, dynamic, base, group_size=2)
    assert output.tolist() == [[[1.0, 2.0], [13.0, 24.0], [35.0, 46.0]]]


def test_input_embedding_scale_composes_with_target_scale():
    args = _args(
        dflash_config={**_args().dflash_config, "input_embedding_scale": 3.0}
    )
    model = DFlash2DraftModel(args)
    target = SimpleNamespace(text_model=SimpleNamespace(embed_scale=2.0))
    target_ops = SimpleNamespace(text_model=lambda model: model.text_model)
    model.bind_target_model(target, target_ops=target_ops)
    assert model.embed_scale == 6.0


def test_selector_uses_predecessor_edges_across_positions():
    selector = CandidateSelector(_args())
    selector.hidden_projection.weight = mx.array([[1.0, 0.0, 0.0, 0.0]])
    selector.predecessor_codebook.weight = mx.array([[1.0], [2.0], [3.0], [4.0]])
    selector.successor_codebook.weight = mx.array([[-2.0], [-1.0], [1.0], [2.0]])
    hidden = mx.array([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    logits = mx.array([[[0.0, 0.0, 5.0, 4.0], [5.0, 4.0, 0.0, 0.0]]])
    path, _, probabilities = selector.select(hidden, logits, mx.array([1]))
    assert path.tolist() == [[3, 1]]
    assert probabilities is None


def test_sparse_rejection_sampling_accepts_full_path():
    draft_tokens = mx.array([[1, 2]], dtype=mx.uint32)
    draft_indices = mx.array([[[1, 0], [2, 0]]], dtype=mx.uint32)
    draft_probs = mx.array([[[1.0, 0.0], [1.0, 0.0]]])
    target_probs = mx.array(
        [[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]
    )
    accepted, bonus = rejection_sample(
        draft_tokens,
        target_probs,
        draft_probs,
        draft_indices,
    )
    assert accepted == 2
    assert bonus == 0


def test_sparse_rejection_sampling_uses_residual_distribution():
    accepted, replacement = rejection_sample(
        mx.array([[0]], dtype=mx.uint32),
        mx.array([[[0.0, 1.0], [1.0, 0.0]]]),
        mx.array([[[1.0]]]),
        mx.array([[[0]]], dtype=mx.uint32),
    )
    assert accepted == 0
    assert replacement == 1


def test_sampling_filters_are_normalized_and_composable():
    probs = sampling_probs(
        mx.array([[4.0, 3.0, 2.0, 1.0]]),
        temperature=1.0,
        top_p=0.6,
        top_k=2,
    )
    assert mx.allclose(probs, mx.array([[1.0, 0.0, 0.0, 0.0]]))
