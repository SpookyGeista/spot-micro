#!/usr/bin/env python3
"""Render orthographic views directly from the published STL geometry."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
fig = plt.figure(figsize=(12, 5), facecolor='white')
views = [
    ('Pi4BPlus_PlateV2.stl', 'Controller plate · top', 90, -90),
    ('L_side_plate.stl', 'Side panel · connector openings', 0, 0),
]
for i, (filename, title, elev, azim) in enumerate(views, 1):
    mesh = trimesh.load(ROOT / 'hardware/modified' / filename)
    ax = fig.add_subplot(1, 2, i, projection='3d', proj_type='ortho')
    light = np.array([.4, -.3, .85]); light /= np.linalg.norm(light)
    intensity = .55 + .45 * np.abs(mesh.face_normals @ light)
    colors = np.array(matplotlib.colors.to_rgb('#58788d'))[None, :] * intensity[:, None]
    ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolors=colors, edgecolors='none', zsort='average'))
    center = mesh.bounds.mean(axis=0)
    radius = max(mesh.extents) * .52
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)
    ax.set_box_aspect((1, 1, 1), zoom=1.25)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, pad=0)
    ax.text2D(.5, .03, filename, transform=ax.transAxes, ha='center', fontsize=10, color='#666666')
fig.subplots_adjust(left=.02, right=.98, top=.90, bottom=.02, wspace=.03, hspace=.02)
fig.savefig(ROOT / 'docs/media/modified-parts.png', dpi=160, facecolor='white')
