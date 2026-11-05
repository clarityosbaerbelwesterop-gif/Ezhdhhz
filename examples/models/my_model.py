"""Your own model code for TITAN: `--model-file examples/models/my_model.py:build`.

The factory receives the vocabulary size (from the tokenizer) and the sequence length, plus any
architecture settings it declares (from --model-config / --set / the profile). It returns an object with
    name, seq, kind = "lm" and build(g, ps, batch, compute) -> (ids_input, logits [batch*seq, vocab]).
Everything inside build() is TITAN graph ops, so autodiff, AdamW, clipping, checkpoints, LoRA,
offload and every backend work unchanged. This example subclasses the configurable transformer
and replaces the MLP with a two-branch mixture (an architecture the config file cannot express).
"""
from titan.nn import TransformerConfig, TransformerLM
from titan.nn.layers import linear


class TwoBranchLM(TransformerLM):
    def _mlp(self, g, ps, x, name, compute):
        c = self.config
        gate = g.sigmoid(linear(g, ps, x, 1, f"{name}.mix", compute=compute))          # per-token mixing weight
        a = linear(g, ps, g.gelu(linear(g, ps, x, c.d_ff, f"{name}.a1", compute=compute)), c.d_model,
                   f"{name}.a2", compute=compute)
        b = linear(g, ps, g.relu(linear(g, ps, x, c.d_ff, f"{name}.b1", compute=compute)), c.d_model,
                   f"{name}.b2", compute=compute)
        ones = g.const([[1.0]], name=f"{name}.one")
        return g.add(g.mul(a, gate), g.mul(b, g.sub(ones, gate)))


def build(vocab, seq, d_model=64, layers=2, heads=4, **_):
    return TwoBranchLM(TransformerConfig(vocab=vocab, seq=seq, d_model=d_model, layers=layers, heads=heads,
                                         norm="rmsnorm", position="rope", name="twobranch"))
