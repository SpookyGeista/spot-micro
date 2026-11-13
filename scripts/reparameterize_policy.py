#!/usr/bin/env python3
"""Transfer existing weights to bounded PPO without refitting the learned gait."""
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from tensordict import TensorDict
from spot_micro.bounded_policy import SquashedActorCritic
from spot_micro.export import export_actor

source=ROOT/"logs/2026-11-11_09-47-54_walk_original_fast_smoothquarter_yaw4p2_continue/model_final.pt"
data=torch.load(ROOT/"work/bounded_policy/teacher_rollouts.pt",weights_only=False)
old=torch.load(source,weights_only=False,map_location="cpu")
old_state=old["model_state_dict"]
cfg=json.loads((source.parent/"robot.json").read_text())
ppo=json.loads((source.parent/"ppo.json").read_text())
ppo["policy"].update(class_name="SquashedActorCritic",init_noise_std=.25,
                      min_noise_std=.05,max_noise_std=.5,latent_mean_bound=4.,action_bound=.99999)
kwargs=copy.deepcopy(ppo["policy"]);kwargs.pop("class_name")
obs=TensorDict({"policy":torch.zeros(1,115),"critic":torch.zeros(1,153)},[1])
policy=SquashedActorCritic(obs,ppo["obs_groups"],12,**kwargs)
missing,unexpected=torch.nn.Module.load_state_dict(policy,{k:v for k,v in old_state.items() if k!="log_std"},strict=False)
assert missing==["noise_logit"] and not unexpected
assert all(torch.equal(policy.state_dict()[k],v) for k,v in old_state.items() if k!="log_std")
torch.set_num_threads(4)
with torch.inference_mode():
    raw=policy.actor(policy.actor_obs_normalizer(data["observations"]))
    delta=policy.action_postprocessor(raw)-raw.clamp(-1,1)
    # Check the whole scalar mapping too, including saturation transitions.
    grid=torch.linspace(-20,20,400001)
    grid_error=(policy.action_postprocessor(grid)-grid.clamp(-1,1)).abs()
    report={"method":"direct weight transfer with near-identity interior mean mapping",
            "source":str(source),"samples":len(raw),"actor_critic_and_normalizers_unchanged":True,
            "action_rmse":float(delta.square().mean().sqrt()),
            "action_mae":float(delta.abs().mean()),"action_error_max":float(delta.abs().max()),
            "scalar_mapping_grid_error_max":float(grid_error.max()),
            "max_desired_joint_difference_rad":float(grid_error.max())*max(cfg["action_scale_rad"]),
            "noise_std_bounds_latent":[.05,.5],"initial_latent_std":.25,
            "old_unbounded_noise_discarded":True,"optimizer_reset":True}
out=ROOT/"logs"/(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+"_bounded_action_reparameterized")
out.mkdir(parents=True)
optimizer=torch.optim.Adam(policy.parameters(),lr=ppo["algorithm"]["learning_rate"])
torch.save({"model_state_dict":policy.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
            "iter":0,"infos":old.get("infos")},out/"model_final.pt")
(out/"ppo.json").write_text(json.dumps(ppo,indent=2)+"\n")
(out/"robot.json").write_text(json.dumps(cfg,indent=2)+"\n")
(out/"conversion.json").write_text(json.dumps(report,indent=2)+"\n")
original_info=json.loads((source.parent/"run.json").read_text())
(out/"run.json").write_text(json.dumps({**original_info,"iterations_requested":0,"conversion":report},indent=2)+"\n")
for path in (ROOT/"spot_micro").glob("*.py"):
    dest=out/"source/spot_micro"/path.name
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,dest)
export_actor(policy,cfg,out/"export",{"conversion_source":str(source),"conversion":report})
(ROOT/"work/bounded_policy/conversion_status.json").write_text(json.dumps({"checkpoint":str(out/"model_final.pt"),"run_dir":str(out),**report},indent=2)+"\n")
print(json.dumps({"checkpoint":str(out/"model_final.pt"),**report},indent=2))
