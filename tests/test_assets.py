import hashlib
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from spot_micro.contract import ROOT, load_robot


class RobotAssetTests(unittest.TestCase):
    def test_pinned_assets_and_joint_mapping(self):
        folder = ROOT / "assets/spot_micro"
        manifest = json.loads((folder / "manifest.json").read_text())
        for item in manifest["files"]:
            self.assertEqual(hashlib.sha256((folder / item["path"]).read_bytes()).hexdigest(), item["sha256"], item["path"])
        root = ET.parse(folder / "spot_micro.urdf").getroot()
        joints = [j for j in root.findall("joint") if j.get("type") != "fixed"]
        cfg = load_robot()
        self.assertEqual({j.get("name") for j in joints}, set(cfg["joint_names"]))
        for mesh in root.findall(".//mesh"):
            self.assertTrue((folder / mesh.get("filename")).is_file())
        total_mass = 0
        for link in root.findall("link"):
            inertia = link.find("inertial")
            mass = float(inertia.find("mass").get("value"))
            total_mass += mass
            self.assertGreater(mass, 0)
            a = {k: float(v) for k, v in inertia.find("inertia").attrib.items()}
            matrix = np.array([[a["ixx"], a["ixy"], a["ixz"]], [a["ixy"], a["iyy"], a["iyz"]], [a["ixz"], a["iyz"], a["izz"]]])
            self.assertTrue((np.linalg.eigvalsh(matrix) > 0).all(), link.get("name"))
        self.assertAlmostEqual(total_mass, 2.96, places=6)

    def test_original_model_coordinates_inertias_and_deployment_limits(self):
        folder = ROOT / "assets/spot_micro_original"
        cfg = load_robot(ROOT / "configs/spotmicro_original.json")
        model = folder / "spot_micro.urdf"
        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), cfg["urdf_sha256"])
        manifest = json.loads((folder / "manifest.json").read_text())
        for item in manifest["files"]:
            self.assertEqual(hashlib.sha256((folder / item["path"]).read_bytes()).hexdigest(), item["sha256"])
        root = ET.parse(model).getroot()
        joints = {j.attrib["name"]: j for j in root.findall("joint") if j.attrib["type"] == "revolute"}
        self.assertEqual(set(joints), set(cfg["joint_names"]))
        for i, name in enumerate(cfg["joint_names"]):
            limits = joints[name].find("limit")
            self.assertGreaterEqual(cfg["joint_lower_rad"][i], float(limits.attrib["lower"]))
            self.assertLessEqual(cfg["joint_upper_rad"][i], float(limits.attrib["upper"]))
        mass = 0
        for link in root.findall("link"):
            inertial = link.find("inertial")
            if inertial is None:
                continue
            mass += float(inertial.find("mass").attrib["value"])
            tensor = inertial.find("inertia")
            diag = np.array([float(tensor.attrib[k]) for k in ("ixx", "iyy", "izz")])
            self.assertTrue((diag > 0).all())
            self.assertLess(float(diag.max()), .01)
            self.assertLessEqual(float(diag.max()), float(diag.sum() - diag.max()) + 1e-12)
        self.assertAlmostEqual(mass, 2.52)
        frame = root.find("joint[@name='original_body_frame']/origin")
        yaw = float(frame.attrib["rpy"].split()[2])
        rotate = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        for leg, x_sign, y_sign in (("front_left", 1, 1), ("front_right", 1, -1),
                                    ("back_left", -1, 1), ("back_right", -1, -1)):
            hip = joints[f"motor_{leg}_hip"]
            position = rotate @ np.array(hip.find("origin").attrib["xyz"].split(), dtype=float)
            self.assertGreater(x_sign * position[0], .08)
            self.assertGreater(y_sign * position[1], .03)
            axis = rotate @ np.array(hip.find("axis").attrib["xyz"].split(), dtype=float)
            np.testing.assert_allclose(axis, [1, 0, 0], atol=1e-12)
        self.assertGreater(cfg["nominal_base_height_m"], .15)
        self.assertLess(cfg["nominal_base_height_m"], .30)


if __name__ == "__main__":
    unittest.main()
