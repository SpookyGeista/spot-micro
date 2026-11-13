"""Bounded gait-quality costs; foot positions/velocities are simulation proxies."""
import math
import torch


def style_terms(contact, clearance, velocity, phase, moving, root_velocity, root_omega,
                previous_contact, previous_vertical_velocity, weights, parameters, dt):
    swing = (phase >= .6) & moving[:, None]
    progress = ((phase-.6)/.4).clamp(0,1)
    target = float(parameters.get("clearance_m",.025))*torch.sin(math.pi*progress)
    target = torch.where(swing,target,torch.zeros_like(target))
    error = clearance-target
    slip_square = velocity[..., :2].square().sum(-1)
    contact_count = contact.sum(-1).clamp(min=1)
    swing_count = swing.sum(-1).clamp(min=1)
    touchdown = contact & ~previous_contact
    downward_speed = torch.maximum((-velocity[...,2]).clamp(min=0),(-previous_vertical_velocity).clamp(min=0))
    wobble = root_omega[:,:2].square().sum(-1)/.7**2 + root_velocity[:,2].square()/.15**2
    terms = {
        "foot_slip": -float(weights.get("foot_slip",0))*((slip_square/.15**2).clamp(max=4)*contact).mean(-1),
        "foot_clearance": -float(weights.get("foot_clearance",0))*((error/.025).square().clamp(max=4)*swing).mean(-1),
        "touchdown": -float(weights.get("touchdown",0))*((downward_speed/.25).square().clamp(max=4)*touchdown).mean(-1)/dt,
        "body_wobble": -float(weights.get("body_wobble",0))*wobble.clamp(max=4),
    }
    metrics = {
        "stance_point_speed_m_s": (slip_square.sqrt()*contact).sum(-1)/contact_count,
        "swing_clearance_error_squared_m2": (error.square()*swing).sum(-1)/swing_count,
        "body_roll_pitch_rate_squared": root_omega[:,:2].square().sum(-1),
        "body_vertical_speed_squared": root_velocity[:,2].square(),
        "touchdown_downward_speed_sum_m_s": (downward_speed*touchdown).sum(-1),
        "touchdown_count": touchdown.sum(-1).float(),
        "stance_point_speed_sum_m_s": (slip_square.sqrt()*contact).sum(-1),
        "stance_count": contact.sum(-1).float(),
        "swing_clearance_error_sum_m2": (error.square()*swing).sum(-1),
        "swing_count": swing.sum(-1).float(),
    }
    return terms, metrics
