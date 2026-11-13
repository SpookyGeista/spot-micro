"""Export actor and its learned normalization, with an explicit I/O contract."""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .contract import metadata


class ExportedActor(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.normalizer = copy.deepcopy(policy.actor_obs_normalizer).cpu().eval()
        self.actor = copy.deepcopy(policy.actor).cpu().eval()
        self.action_postprocessor = copy.deepcopy(getattr(policy, "action_postprocessor", torch.nn.Identity())).cpu().eval()

    def forward(self, observation):
        return self.action_postprocessor(self.actor(self.normalizer(observation)))


def export_actor(policy, cfg, destination, provenance=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    info = metadata(cfg)
    info["training"] = provenance or {}
    info["action_distribution"] = "squashed_normal" if getattr(policy, "bounded_actions", False) else "normal_clipped_by_runtime"
    model = ExportedActor(policy).eval()
    sample = torch.zeros(1, info["observation_size"])
    with torch.inference_mode():
        traced = torch.jit.trace(model, sample)
        traced.save(str(destination / "policy.pt"))
        torch.onnx.export(model, sample, str(destination / "policy.onnx"),
                          input_names=["observation"], output_names=["action"],
                          dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
                          opset_version=17, dynamo=False)
    (destination / "metadata.json").write_text(json.dumps(info, indent=2) + "\n")
    # Verify the actual ONNX artifact, including empirical normalization.
    import onnxruntime as ort
    session = ort.InferenceSession(str(destination / "policy.onnx"), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(2026)
    observations = rng.normal(0, 0.5, (8, info["observation_size"])).astype(np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(observations)).numpy()
    actual = session.run(None, {"observation": observations})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    report = {"onnx_max_abs_error": float(np.max(np.abs(actual - expected))),
              "actor_parameters": sum(p.numel() for p in model.actor.parameters()),
              "input_shape": [1, info["observation_size"]], "output_shape": [1, 12]}
    (destination / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
