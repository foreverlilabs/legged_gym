import importlib.util
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_quadruped_urdf.py"
SPEC = importlib.util.spec_from_file_location("prepare_quadruped_urdf", str(SCRIPT))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


INERTIAL = """
    <inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
"""


class PrepareQuadrupedUrdfTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        package = self.root / "cad_robot"
        (package / "urdf").mkdir(parents=True)
        (package / "meshes").mkdir()
        (package / "meshes" / "body.STL").write_text("solid body\nendsolid body\n")
        parts = ['<robot name="cad_robot">']
        parts.append('<link name="body"><visual><geometry><mesh filename="package://cad_robot/meshes/body.STL"/></geometry></visual><collision><geometry><mesh filename="package://cad_robot/meshes/body.STL"/></geometry></collision>{}</link>'.format(INERTIAL))
        mapping_legs = {}
        for leg in MODULE.LEG_ORDER:
            parent = "body"
            mapping_legs[leg] = {}
            for index, role in enumerate(MODULE.ROLE_ORDER):
                joint = "{}_cad_joint_{}".format(leg, index)
                child = "{}_cad_link_{}".format(leg, index)
                parts.append('<joint name="{}" type="revolute"><parent link="{}"/><child link="{}"/><axis xyz="0 2 0"/><limit lower="-2" upper="2" effort="40" velocity="20"/></joint>'.format(joint, parent, child))
                parts.append('<link name="{}">{}</link>'.format(child, INERTIAL))
                mapping_legs[leg][role + "_joint"] = joint
                parent = child
            mapping_legs[leg]["foot_link"] = ""
            mapping_legs[leg]["foot_origin"] = [0, 0, -0.2]
        parts.append("</robot>")
        self.urdf = package / "urdf" / "cad_robot.urdf"
        self.urdf.write_text("\n".join(parts))
        self.mapping = self.root / "mapping.json"
        self.mapping.write_text(json.dumps({
            "robot_name": "test_dog",
            "standard": "a1",
            "base_link": "body",
            "legs": mapping_legs,
            "joint_limits": {"hip": {}, "thigh": {}, "calf": {}},
            "joint_dynamics": {"damping": 0, "friction": 0},
            "foot": {"radius": 0.025, "mass": 0.03, "replace_collision": True},
        }))

    def tearDown(self):
        self.temp.cleanup()

    def test_convert_and_validate_a1(self):
        output = self.root / "output"
        MODULE.convert(self.urdf, self.mapping, output, None, False, False)
        converted = output / "urdf" / "test_dog.urdf"
        root = ET.parse(str(converted)).getroot()
        joint_names = {joint.get("name") for joint in root.findall("joint")}
        link_names = {link.get("name") for link in root.findall("link")}
        self.assertIn("FL_hip_joint", joint_names)
        self.assertIn("RR_calf_joint", joint_names)
        self.assertIn("RL_foot", link_names)
        foot_joint = root.find("./joint[@name='FR_foot_fixed']")
        self.assertEqual("true", foot_joint.get("dont_collapse"))
        self.assertIsNotNone(root.find("./link[@name='FL_foot']/collision/geometry/sphere"))
        mesh = root.find(".//mesh")
        self.assertEqual("../meshes/body.STL", mesh.get("filename"))
        self.assertTrue((output / "meshes" / "body.STL").is_file())
        errors, unused_warnings = MODULE.validate_standard(root, "a1", converted)
        self.assertEqual([], errors)

    def test_template_finds_root_and_canonical_aliases(self):
        template = MODULE.build_template(self.urdf, "a1")
        self.assertEqual("body", template["base_link"])
        self.assertEqual("a1", template["standard"])


if __name__ == "__main__":
    unittest.main()
