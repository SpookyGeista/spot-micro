"""Read support vertices of the lower leg and its fixed toe from the URDF."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def origin(element):
    transform=np.eye(4)
    if element is not None:
        xyz=np.fromstring(element.get('xyz','0 0 0'),sep=' ')
        rpy=np.fromstring(element.get('rpy','0 0 0'),sep=' ')
        transform=trimesh.transformations.euler_matrix(*rpy)
        transform[:3,3]=xyz
    return transform


def lower_leg_vertices(urdf_path):
    path=Path(urdf_path)
    root=ET.parse(path).getroot()
    links={x.get('name'):x for x in root.findall('link')}
    joints=root.findall('joint')
    result=[]
    for leg in ('front_left','front_right','back_left','back_right'):
        name=leg+'_lower_leg'
        group=[(name,np.eye(4))]
        for current,transform in group:
            for joint in joints:
                if joint.get('type')=='fixed' and joint.find('parent').get('link')==current:
                    group.append((joint.find('child').get('link'),transform@origin(joint.find('origin'))))
        points=[]
        for link,transform in group:
            for collision in links[link].findall('collision'):
                geometry=collision.find('geometry')
                box=geometry.find('box');mesh=geometry.find('mesh')
                if box is not None:
                    vertices=trimesh.creation.box(np.fromstring(box.get('size'),sep=' ')).vertices
                elif mesh is not None:
                    loaded=trimesh.load_mesh(path.parent/mesh.get('filename'))
                    vertices=loaded.convex_hull.vertices*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                else:
                    raise ValueError('Unsupported collision shape in '+link)
                pose=transform@origin(collision.find('origin'))
                points.append(vertices@pose[:3,:3].T+pose[:3,3])
        result.append(np.concatenate(points))
    return result


def mesh_clearance(positions,quaternions,vertices,ground_z=0.):
    """Lowest collision-geometry point above a flat plane; wxyz quaternions."""
    w,x,y,z=np.moveaxis(quaternions,-1,0)
    up_local=np.stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),axis=-1)
    support=np.stack([np.min(up_local[...,leg,:]@points.T,axis=-1) for leg,points in enumerate(vertices)],axis=-1)
    ground=np.asarray(ground_z)
    if ground.shape != positions.shape[:-1]:
        ground=ground[...,None]
    return np.maximum(positions[...,2]+support-ground,0.)
