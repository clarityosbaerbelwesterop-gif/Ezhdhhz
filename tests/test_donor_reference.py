"""The streamed Qwen3-MoE layer forward must equal transformers' reference model (random weights)."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_qwen3moe_forward_equals_transformers_reference():
    from titan.donor import qwen3moe as Q
    if not hasattr(transformers, "Qwen3MoeForCausalLM"):
        pytest.skip("transformers without Qwen3-MoE")
    torch.manual_seed(0)
    hc = transformers.Qwen3MoeConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                                     num_attention_heads=8, num_key_value_heads=2, head_dim=16, num_experts=12,
                                     num_experts_per_tok=4, moe_intermediate_size=24, rms_norm_eps=1e-6, rope_theta=1e7,
                                     norm_topk_prob=True, max_position_embeddings=512, decoder_sparse_step=1,
                                     mlp_only_layers=[])
    ref = transformers.Qwen3MoeForCausalLM(hc).eval().float()
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(torch.randn_like(p) * (0.2 if p.ndim > 1 else 1.0) + (1.0 if p.ndim == 1 else 0.0))
    sd = ref.state_dict()
    cfg = Q.QwenMoeConfig(64, 3, 8, 2, 16, 12, 4, 24, 1e-6, 1e7, 97)
    lens = [17, 9, 23]
    ids = [torch.randint(0, 97, (n,)) for n in lens]
    with torch.no_grad():
        refs = [ref(i[None]).logits[0] for i in ids]
        fb = Q.FlatBatch(lens)
        h = sd["model.embed_tokens.weight"][torch.cat(ids)]
        cos, sin = Q.rope_tables(max(lens), cfg.head_dim, cfg.rope_theta)
        for layer in range(cfg.layers):
            p = f"model.layers.{layer}."
            W = {k: sd[p + v] for k, v in Q.LAYER_KEYS.items()}
            if p + "mlp.experts.gate_up_proj" in sd:                       # fused expert storage
                gu, f = sd[p + "mlp.experts.gate_up_proj"], cfg.moe_inter
                W["e_gate"], W["e_up"], W["e_down"] = gu[:, :f], gu[:, f:], sd[p + "mlp.experts.down_proj"]
            else:
                for k, v in Q.EXPERT_KEYS.items():
                    W[k] = torch.stack([sd[f"{p}mlp.experts.{e}.{v}"] for e in range(cfg.experts)])
            h = fb.attend(h, W, cfg, cos, sin, batch=2)                    # padded batches of unequal lengths
            h, info = Q.moe(h, W, cfg, groups=fb.seq % 2, n_groups=2)
            assert float(info["stats"]["count"].sum()) == sum(lens) * cfg.top_k
        lg = torch.nn.functional.linear(Q.rms_norm(h, sd["model.norm.weight"], cfg.eps), sd["lm_head.weight"])
    for i in range(3):
        mine = lg[fb.starts[i]:fb.starts[i + 1]]
        assert float((mine - refs[i]).abs().max()) < 2e-4 * max(1.0, float(refs[i].abs().max()))


def test_mistral_large4_forward_equals_transformers_reference():
    """Layer-streamed Mistral Large 4 text stack (LeRoPE, NoPE layers, key sharing, scalable softmax, dense first
    layer, routed + shared experts) == transformers' MistralLarge4TextModel on random weights."""
    if not hasattr(transformers, "MistralLarge4TextModel"):
        pytest.skip("transformers without Mistral Large 4")
    from titan.donor import mistral_l4 as M
    torch.manual_seed(0)
    L = 8
    hc = transformers.MistralLarge4TextConfig(
        vocab_size=101, hidden_size=48, intermediate_size=96, moe_intermediate_size=40, num_hidden_layers=L,
        num_attention_heads=4, num_key_value_heads=4, n_routed_experts=10, num_experts_per_tok=3, kv_lora_rank=12,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16, index_topk=64, index_head_dim=8, index_n_heads=2,
        kv_source_layers=[0, 1, 2, 3, 4, 5, 6, 3], max_position_embeddings=256,
        rope_parameters={"rope_type": "yarn", "rope_theta": 10000.0, "factor": 8, "original_max_position_embeddings": 32,
                         "beta_fast": 32, "beta_slow": 1, "attention_factor": 1.0})
    ref = transformers.MistralLarge4TextModel(hc).eval().float()
    with torch.no_grad():
        for n, p in ref.named_parameters():
            p.copy_(torch.randn_like(p) * (0.2 if p.ndim > 1 else 0.3) + (1.0 if p.ndim == 1 and "norm" in n else 0.0))
        ref.rotary_emb.alpha.copy_(torch.randn_like(ref.rotary_emb.alpha) * 0.3)
    sd = ref.state_dict()
    cfg = M.ML4Config.from_hf_text(hc.to_dict())
    assert cfg.use_rope == [1, 1, 1, 0, 1, 1, 1, 0] and cfg.kv_source[7] == 3 and cfg.mlp_types[0] == "dense"
    lens = [19, 7, 26]
    ids = [torch.randint(0, 101, (n,)) for n in lens]
    with torch.no_grad():
        refs = [ref(input_ids=i[None]).last_hidden_state[0] for i in ids]
        fb = M.FlatBatch(lens)
        h = sd["embed_tokens.weight"][torch.cat(ids)]
        inv_freq, scaling = M.rope_frequencies(hc.to_dict())            # the tool's own path to the frequencies
        assert torch.equal(inv_freq, ref.rotary_emb.inv_freq) and scaling == ref.rotary_emb.attention_scaling
        cos, sin = M.rope_tables(max(lens), inv_freq, sd["rotary_emb.alpha"], scaling)
        shared = {}
        for layer in range(L):
            p = f"layers.{layer}."
            W = {"in_ln": sd[p + "input_layernorm.weight"], "post_ln": sd[p + "post_attention_layernorm.weight"],
                 "wq": sd[p + "self_attn.q_proj.weight"], "wo": sd[p + "self_attn.o_proj.weight"],
                 "wkv_b": sd[p + "self_attn.kv_b_proj.weight"], "ss_s": sd[p + "self_attn.softmax_scaling.scale"],
                 "ss_b": sd[p + "self_attn.softmax_scaling.bias"]}
            if cfg.kv_source[layer] == layer:
                W["wkv_a"], W["kv_a_norm"] = sd[p + "self_attn.kv_a_proj_with_mqa.weight"], sd[p + "self_attn.kv_a_layernorm.weight"]
            h, keys = fb.attend(h, W, cfg, layer, cos, sin, shared.get(cfg.kv_source[layer]), batch=2)
            if keys is not None:
                shared[layer] = keys
            xn = M.rms_norm(h, W["post_ln"], cfg.eps)
            if cfg.mlp_types[layer] == "dense":
                h = h + M.swiglu(xn, sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"], sd[p + "mlp.down_proj.weight"])
                continue
            topi, topw = M.route(xn, sd[p + "mlp.gate.weight"], cfg)
            out = M.swiglu(xn, sd[p + "mlp.shared_experts.gate_proj.weight"], sd[p + "mlp.shared_experts.up_proj.weight"],
                           sd[p + "mlp.shared_experts.down_proj.weight"])
            gu, dn, f = sd[p + "mlp.experts.gate_up_proj"], sd[p + "mlp.experts.down_proj"], cfg.moe_inter
            for e in range(cfg.experts):                                  # one expert at a time, as when streaming
                tok, slot = torch.where(topi == e)
                if len(tok):
                    out.index_add_(0, tok, M.swiglu(xn[tok], gu[e, :f], gu[e, f:], dn[e]) * topw[tok, slot, None])
            h = h + out
        h = M.rms_norm(h, sd["norm.weight"], cfg.eps)
    for i in range(3):
        mine = h[fb.starts[i]:fb.starts[i + 1]]
        assert float((mine - refs[i]).abs().max()) < 3e-4 * max(1.0, float(refs[i].abs().max()))


def test_fp8_block_dequantisation_matches_torch_and_table():
    import numpy as np
    from titan.donor import fp8, mistral_l4 as M
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 256, (256, 384), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0                                     # no NaN codes
    scale = (2.0 ** rng.integers(-6, 3, (2, 3))).astype(np.float32)
    w = M.dequant(codes.tobytes(), (256, 384), scale.tobytes())
    table = fp8.apply_block_scale(fp8.decode_fp8(codes.tobytes()).reshape(256, 384), scale)
    assert np.array_equal(w.numpy(), table)                               # torch float8_e4m3fn == our table, exactly
    assert np.array_equal(fp8.encode_e4m3(fp8.decode_fp8(codes.tobytes())), codes.ravel() | 0)  # decode/encode round trip
