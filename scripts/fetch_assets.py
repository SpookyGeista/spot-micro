#!/usr/bin/env python3
"""Fetch only the URDF and meshes we need, pinned to an upstream commit."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets/spot_micro"
REPO = "OpenQuadruped/spot_mini_mini"
COMMIT = "c7e4905be176c63fa0e68a09c177b937e916fa60"
PREFIX = "spotmicro/util/pybullet_data/assets/urdf/"


def download(remote: str, local: Path) -> dict:
    url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{remote}"
    data = urllib.request.urlopen(url, timeout=90).read()
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    return {"path": str(local.relative_to(DEST)), "upstream_path": remote, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    records = [download(PREFIX + "spot.urdf", DEST / "upstream.urdf")]
    tree = ET.parse(DEST / "upstream.urdf")
    filenames = sorted({m.attrib["filename"].split("package://mini_ros/")[1] for m in tree.findall(".//mesh")})
    jobs = [(PREFIX + name, DEST / name) for name in filenames]
    jobs.extend([(PREFIX + "stl/LICENSE.txt", DEST / "stl/LICENSE.txt"), (PREFIX + "stl/README.txt", DEST / "stl/README.txt"), ("LICENSE", DEST / "LICENSE")])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(lambda args: download(*args), jobs):
            print(record["path"], record["bytes"], flush=True)
            records.append(record)
    # Preserve the unmodified upstream model; create an Isaac-compatible copy.
    root = tree.getroot()
    root.attrib["name"] = "spot_micro"
    for mesh in root.findall(".//mesh"):
        mesh.attrib["filename"] = mesh.attrib["filename"].replace("package://mini_ros/", "")
    for joint in root.findall("joint"):
        if joint.attrib["type"] == "revolute":
            # Upstream has placeholder 1000 Nm effort and 0.7 rad/s velocity.
            joint.find("limit").set("effort", "1.2")
            joint.find("limit").set("velocity", "12.0")
    for inertia in root.findall(".//inertia"):
        for key in ("iyx", "izx", "izy"):
            inertia.attrib.pop(key, None)
    ET.indent(tree, space="  ")
    tree.write(DEST / "spot_micro.urdf", encoding="utf-8", xml_declaration=True)
    manifest = {"repository": f"https://github.com/{REPO}", "commit": COMMIT, "files": records, "changes": ["Resolve ROS package mesh paths to local relative paths", "Replace placeholder joint effort/velocity bounds; actual servo envelope is applied in ServoDynamics", "Remove nonstandard duplicate inertia attributes"]}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Robot assets ready: {DEST}")


if __name__ == "__main__":
    main()
