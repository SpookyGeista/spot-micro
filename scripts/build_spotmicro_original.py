#!/usr/bin/env python3
"""Fetch original SpotMicro meshes and build a provisional, explicitly audited dynamics model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets/spot_micro_original"
COMMIT = "bf91ab37620cbb036968952da92773522361aad7"
PREFIX = "spotmicro/util/pybullet_data/assets/urdf/"
BASE = f"https://raw.githubusercontent.com/OpenQuadruped/spot_mini_mini/{COMMIT}/"


def fetch(source, target):
    data = urllib.request.urlopen(BASE + source, timeout=90).read()
    path = DEST / target
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": target, "source": source, "sha256": hashlib.sha256(data).hexdigest()}


def rotation(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def vector(text):
    return np.array([float(x) for x in text.split()])


def mesh_bounds(mesh):
    data = (DEST / mesh.attrib["filename"]).read_bytes()
    count = int.from_bytes(data[80:84], "little")
    if len(data) == 84 + 50 * count:
        dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
        vertices = np.frombuffer(data, dtype=dtype, count=count, offset=84)["vertices"].reshape(-1, 3)
    else:
        vertices = np.array([line.split()[1:] for line in data.decode().splitlines()
                             if line.strip().startswith("vertex ")], dtype=float)
    vertices = vertices * vector(mesh.attrib.get("scale", "1 1 1"))
    return vertices.min(axis=0), vertices.max(axis=0)


def collision_bounds(link):
    collision = link.find("collision")
    geometry = collision.find("geometry")
    if geometry.find("box") is not None:
        size = vector(geometry.find("box").attrib["size"])
        lo, hi = -size / 2, size / 2
    else:
        lo, hi = mesh_bounds(geometry.find("mesh"))
    origin = collision.find("origin")
    xyz = vector(origin.attrib.get("xyz", "0 0 0")) if origin is not None else np.zeros(3)
    rpy = vector(origin.attrib.get("rpy", "0 0 0")) if origin is not None else np.zeros(3)
    return lo, hi, xyz, rpy


def transforms(root, positions):
    result = {"base_link": np.eye(4)}
    pending = list(root.findall("joint"))
    while pending:
        available = [j for j in pending if j.find("parent").attrib["link"] in result]
        if not available:
            raise ValueError("URDF is not a connected tree")
        for joint in available:
            origin = joint.find("origin")
            transform = np.eye(4)
            if origin is not None:
                transform[:3, :3] = rotation(vector(origin.attrib.get("rpy", "0 0 0")))
                transform[:3, 3] = vector(origin.attrib.get("xyz", "0 0 0"))
            if joint.attrib["type"] == "revolute":
                axis = vector(joint.find("axis").attrib["xyz"])
                q = positions[joint.attrib["name"]]
                skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                transform[:3, :3] = transform[:3, :3] @ (np.eye(3) + math.sin(q)*skew + (1-math.cos(q))*(skew@skew))
            result[joint.find("child").attrib["link"]] = result[joint.find("parent").attrib["link"]] @ transform
            pending.remove(joint)
    return result


def build(download=False):
    records = []
    if download:
        records.append(fetch(PREFIX + "spot.urdf", "upstream.urdf"))
        upstream = ET.parse(DEST / "upstream.urdf").getroot()
        names = sorted({n.attrib["filename"] for n in upstream.findall(".//mesh")})
        jobs = [(PREFIX + n, n) for n in names]
        jobs += [(PREFIX + "stl/LICENSE.txt", "stl/LICENSE.txt"), (PREFIX + "stl/README.txt", "stl/README.txt"), ("LICENSE", "LICENSE")]
        with ThreadPoolExecutor(max_workers=4) as pool:
            records.extend(pool.map(lambda pair: fetch(*pair), jobs))
    root = ET.parse(DEST / "upstream.urdf").getroot()
    root.attrib["name"] = "spot_micro_original"
    names = {"base_link": "body_shell"}
    joints = {}
    for old_leg, new_leg in (("front_left", "front_left"), ("front_right", "front_right"),
                             ("rear_left", "back_left"), ("rear_right", "back_right")):
        for old_part, new_part in (("shoulder_link", "hip"), ("leg_link", "upper_leg"),
                                   ("foot_link", "lower_leg"), ("toe_link", "toe"),
                                   ("leg_link_cover", "cover")):
            names[f"{old_leg}_{old_part}"] = f"{new_leg}_{new_part}"
        joints[f"motor_{old_leg}_shoulder"] = f"motor_{new_leg}_hip"
        joints[f"motor_{old_leg}_leg"] = f"motor_{new_leg}_upper_leg"
        joints[f"foot_motor_{old_leg}"] = f"motor_{new_leg}_lower_leg"
    assumptions = []
    for link in root.findall("link"):
        link.attrib["name"] = names.get(link.attrib["name"], link.attrib["name"])
        inertial = link.find("inertial")
        if inertial is None:
            continue  # Covers remain visual-only and are merged with their parent.
        mass = float(inertial.find("mass").attrib["value"])
        lo, hi, xyz, rpy = collision_bounds(link)
        size = hi - lo
        center = xyz + rotation(rpy) @ ((lo + hi) / 2)
        inertia = mass / 12 * np.array([size[1]**2 + size[2]**2, size[0]**2 + size[2]**2, size[0]**2 + size[1]**2])
        origin = inertial.find("origin")
        if origin is None:
            origin = ET.SubElement(inertial, "origin")
        origin.attrib.update(xyz=" ".join(map(str, center)), rpy=" ".join(map(str, rpy)))
        inertial.find("inertia").attrib.clear()
        inertial.find("inertia").attrib.update(ixx=str(inertia[0]), iyy=str(inertia[1]), izz=str(inertia[2]), ixy="0", ixz="0", iyz="0")
        assumptions.append({"link": link.attrib["name"], "mass_kg": mass,
                            "method": "uniform collision-bounding box; estimated, not CAD/measured mass distribution",
                            "box_m": size.tolist(), "com_in_link_m": center.tolist()})
    for joint in root.findall("joint"):
        for tag in ("parent", "child"):
            element = joint.find(tag)
            element.attrib["link"] = names.get(element.attrib["link"], element.attrib["link"])
        if joint.attrib["type"] == "revolute":
            joint.attrib["name"] = joints[joint.attrib["name"]]
            axis = joint.find("axis")
            axis.attrib["xyz"] = " ".join(map(str, -vector(axis.attrib["xyz"])))
            limit = joint.find("limit")
            low, high = float(limit.attrib["lower"]), float(limit.attrib["upper"])
            limit.attrib.update(lower=str(-high), upper=str(-low), effort="1.2", velocity="12.0")
            dynamics = joint.find("dynamics")
            if dynamics is not None:
                dynamics.attrib.update(damping="0", friction="0")
    # Original labels use front=-X and left=-Y. Rotate the whole geometry,
    # then reverse all joint coordinates to retain +X forward/+Y left and our
    # existing hip/upper/knee conventions. The fixed root is merged by Isaac.
    root.insert(0, ET.Element("link", name="base_link"))
    joint = ET.SubElement(root, "joint", name="original_body_frame", type="fixed")
    ET.SubElement(joint, "parent", link="base_link")
    ET.SubElement(joint, "child", link="body_shell")
    ET.SubElement(joint, "origin", xyz="0 0 0", rpy=f"0 0 {math.pi}")
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    output = DEST / "spot_micro.urdf"
    tree.write(output, encoding="utf-8", xml_declaration=True)
    cfg = json.loads((ROOT / "configs/robot.json").read_text())
    poses = transforms(root, dict(zip(cfg["joint_names"], cfg["rest_positions_rad"])))
    minimum_z = float("inf")
    for link in root.findall("link"):
        if link.find("collision") is None:
            continue
        lo, hi, xyz, rpy = collision_bounds(link)
        corners = np.array(list(itertools.product(*zip(lo, hi))))
        local = (rotation(rpy) @ corners.T).T + xyz
        world = (poses[link.attrib["name"]] @ np.column_stack((local, np.ones(8))).T).T
        minimum_z = min(minimum_z, float(world[:, 2].min()))
    cfg.update(description="Original SpotMicro MG996R-family geometry; mass and bounding-box inertias are provisional, not hardware-identified.",
               model_family="spotmicro_original", urdf_asset_path="assets/spot_micro_original/spot_micro.urdf",
               urdf_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
               nominal_base_height_m=round(-minimum_z + .003, 6),
               reward_weights={"track_velocity": 8.0, "track_yaw": 1.4})
    (ROOT / "configs/spotmicro_original.json").write_text(json.dumps(cfg, indent=2) + "\n")
    manifest = {"repository": "https://github.com/OpenQuadruped/spot_mini_mini", "commit": COMMIT,
                "confirmed_print_source": "https://www.thingiverse.com/thing:3445283",
                "geometry_match": "Original SpotMicro family; exact servo-horn/CAD variant not independently verified",
                "mass_kg": sum(a["mass_kg"] for a in assumptions), "inertia_assumptions": assumptions,
                "converted_urdf_sha256": cfg["urdf_sha256"], "nominal_base_height_m": cfg["nominal_base_height_m"],
                "changes": ["Replace placeholder inertia with mass-scaled collision bounding-box estimates and their COM",
                            "Normalize body axes with 180-degree fixed root rotation; reverse joint axes/limits and rename joints/links",
                            "Replace nonphysical joint effort/velocity limits; remove source Coulomb friction placeholders (servo dynamics remain active)"],
                "hardware_validated": False}
    if not records and (DEST / "manifest.json").exists():
        records = json.loads((DEST / "manifest.json").read_text()).get("files", [])
    manifest["files"] = records
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"model": str(output), "mass_kg": manifest["mass_kg"],
                      "rest_height_m": cfg["nominal_base_height_m"], "reward_weights": cfg["reward_weights"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    build(parser.parse_args().download)
