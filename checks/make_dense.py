"""Dense 2-layer champion stand-in at the seed's real width, saved with the team's save_checkpoint; real token ids."""
import copy, json, sys
team, repo, handoff, out = sys.argv[1:5]
sys.path.insert(0, team); sys.path.insert(0, repo)
import numpy as np
from accel import forge_io as FIO, model as M
from accel.config import ModelConfig
from titan.donor import st
from titan.donor.vocab import SeedBPE
state = json.load(open(f"{handoff}/mirror/workspace/gpu-runs/s2-run2-b200/fetch-vol/step-007500/state.json"))
state = copy.deepcopy(state); state["config"]["model"]["n_layers"] = 2; state.pop("model_sha256", None)
base = ModelConfig.from_dict(state["config"]["model"])
p = M.init_params(base, seed=3, exact=False)
rd = st.LocalReader(f"{handoff}/weights_subset/champion_step007500_WEIGHT_SUBSET.safetensors"); refs, _ = st.tensor_refs(rd, "s")
real = lambda n: np.frombuffer(st.fetch(rd, refs[n]), "<f4").reshape(refs[n].shape).copy()
p["tok_emb.weight"], p["norm.weight"] = real("tok_emb.weight"), real("norm.weight")
for i, s in ((0, 11), (1, 23)):
    p[f"blocks.{i}.attn_norm.weight"], p[f"blocks.{i}.ffn_norm.weight"] = real(f"blocks.{s}.attn_norm.weight"), real(f"blocks.{s}.ffn_norm.weight")
z = {k: np.zeros_like(v) for k, v in p.items()}
sha = FIO.save_checkpoint(f"{out}/dense", 7500, p, z, z, state)
tok = SeedBPE(f"{handoff}/mirror/workspace/v3-frozen/base-v3/tokenizer.json")
chunks = json.load(open(f"{out}/../calib_chunks.json"))
ids = np.stack([np.array(tok.encode(chunks[i]["text"])[:96]) for i in (0, 60)]).astype(np.int32)
np.save(f"{out}/check_ids.npy", ids)
print("dense saved", sha, "params", sum(v.size for v in p.values()), "ids", ids.shape)
