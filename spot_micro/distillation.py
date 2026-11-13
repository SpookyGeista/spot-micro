"""Explicit observation projection for teachers and longer sensor histories."""
import copy
import torch
from .contract import frame_fields, frame_size


def sensor_projection(teacher_cfg,student_cfg):
    if student_cfg['history_length']<teacher_cfg['history_length']:
        raise ValueError('Student history must contain every teacher sensor frame')
    fields={};offset=0
    for name,size in frame_fields(student_cfg):
        fields[name]=(offset,size);offset+=size
    indices=[]
    for frame in range(student_cfg['history_length']-teacher_cfg['history_length'],student_cfg['history_length']):
        for name,size in frame_fields(teacher_cfg):
            if name not in fields or fields[name][1]!=size:
                raise ValueError('Teacher field missing from sensor history: '+name)
            start=frame*frame_size(student_cfg)+fields[name][0]
            indices.extend(range(start,start+size))
    return indices


def initialize_student(student,teacher,teacher_cfg,student_cfg):
    source=teacher.state_dict()
    state=copy.deepcopy(student.state_dict())
    indices=sensor_projection(teacher_cfg,student_cfg)
    student_size=frame_size(student_cfg)*student_cfg['history_length']
    teacher_size=frame_size(teacher_cfg)*teacher_cfg['history_length']
    critic_indices=indices+list(range(student_size,student_size+38))
    for key,value in source.items():
        if key=='noise_logit':
            continue
        if key in ('actor.0.weight','critic.0.weight'):
            selected=indices if key.startswith('actor') else critic_indices
            state[key].zero_()
            state[key][:,selected]=value[:,:teacher_size] if key.startswith('actor') else value
        elif 'normalizer' in key and value.ndim==2:
            # Existing sensor moments repeat for older history frames. New
            # acceleration channels retain zero mean/unit variance after scaling.
            target=state[key]
            source_sensor=value[:,:teacher_size]
            old_field_offsets={};offset=0
            for name,size in frame_fields(teacher_cfg):
                old_field_offsets[name]=(offset,size);offset+=size
            for frame in range(student_cfg['history_length']):
                old_frame=max(0,frame-student_cfg['history_length']+teacher_cfg['history_length'])
                offset=frame*frame_size(student_cfg)
                for name,size in frame_fields(student_cfg):
                    if name in old_field_offsets:
                        old_start=old_frame*frame_size(teacher_cfg)+old_field_offsets[name][0]
                        target[:,offset:offset+size]=source_sensor[:,old_start:old_start+size]
                    offset+=size
            if key.startswith('critic'):
                target[:,student_size:]=value[:,teacher_size:]
        else:
            state[key]=value.detach().clone()
    student.load_state_dict(state)
    return indices


def teacher_observations(observations,indices):
    return {'critic':torch.cat((observations['policy'][:,indices],observations['critic'][:,-38:]),dim=-1)}
