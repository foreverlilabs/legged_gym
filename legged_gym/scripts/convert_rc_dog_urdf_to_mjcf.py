"""Convert the RC-Dog training URDF into a MuJoCo MJCF model.

The converter deliberately uses the collision primitives from the training URDF
and keeps the 12 actuated joints in the URDF order (FR, FL, RR, RL).  It has no
third-party dependency, so it can be run before installing the MuJoCo Python
package.
"""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def _floats(text: str, count: int | None = None) -> list[float]:
    values = [float(v) for v in text.split()]
    if count is not None and len(values) != count:
        raise ValueError(f"expected {count} values, got {len(values)}: {text!r}")
    return values


def _fmt(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def _indent(element: ET.Element, level: int = 0) -> None:
    """Pretty-print XML on Python versions before ElementTree.indent."""
    padding = "\n" + "  " * level
    if len(element):
        if not element.text or not element.text.strip():
            element.text = padding + "  "
        for child in element:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = padding
    if level and (not element.tail or not element.tail.strip()):
        element.tail = padding


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _joint_link(joint: ET.Element, tag: str) -> str:
    element = joint.find(tag)
    if element is None or "link" not in element.attrib:
        raise ValueError(f"joint {joint.attrib.get('name', '<unnamed>')} has no {tag} link")
    return element.attrib["link"]


def _origin(parent: ET.Element, tag: str = "origin") -> tuple[list[float], list[float]]:
    origin = parent.find(tag)
    if origin is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return _floats(origin.attrib.get("xyz", "0 0 0"), 3), _floats(origin.attrib.get("rpy", "0 0 0"), 3)


def _rpy_matrix(rpy: list[float]) -> list[list[float]]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _rotate_inertia(values: list[float], rpy: list[float]) -> list[float]:
    """Rotate a URDF inertia tensor into the link frame for MuJoCo."""
    ixx, iyy, izz, ixy, ixz, iyz = values
    tensor = [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]
    rotation = _rpy_matrix(rpy)
    rotated = [[sum(rotation[i][k] * tensor[k][l] * rotation[j][l] for k in range(3) for l in range(3))
                for j in range(3)] for i in range(3)]
    return [rotated[0][0], rotated[1][1], rotated[2][2], rotated[0][1], rotated[0][2], rotated[1][2]]


def _add_inertial(body: ET.Element, link: ET.Element) -> None:
    inertial = link.find("inertial")
    if inertial is None:
        return
    pos, inertial_rpy = _origin(inertial)
    mass = inertial.find("mass")
    inertia = inertial.find("inertia")
    if mass is None or inertia is None:
        return
    attrs = {
        "pos": _fmt(pos),
        "mass": mass.attrib["value"],
        "fullinertia": _fmt(_rotate_inertia([
            float(inertia.attrib["ixx"]), float(inertia.attrib["iyy"]), float(inertia.attrib["izz"]),
            float(inertia.attrib.get("ixy", 0.0)), float(inertia.attrib.get("ixz", 0.0)),
            float(inertia.attrib.get("iyz", 0.0)),
        ], inertial_rpy)),
    }
    ET.SubElement(body, "inertial", attrs)


def _mesh_file(geometry: ET.Element) -> str | None:
    mesh = geometry.find("mesh")
    return None if mesh is None else Path(mesh.attrib["filename"]).name


def _add_geoms(body: ET.Element, link: ET.Element, mesh_names: dict[str, str]) -> None:
    visual_index = 0
    for visual in link.findall("visual"):
        geometry = visual.find("geometry")
        if geometry is None:
            continue
        filename = _mesh_file(geometry)
        if filename is None:
            continue
        pos, euler = _origin(visual)
        key = filename
        mesh_name = mesh_names.setdefault(key, f"mesh_{_safe_name(Path(filename).stem)}")
        ET.SubElement(body, "geom", {
            "name": f"{link.attrib['name']}_visual_{visual_index}",
            "type": "mesh", "mesh": mesh_name,
            "pos": _fmt(pos), "euler": _fmt(euler),
            "contype": "0", "conaffinity": "0", "rgba": "0.72 0.74 0.78 1",
        })
        visual_index += 1

    collision_index = 0
    for collision in link.findall("collision"):
        geometry = collision.find("geometry")
        if geometry is None:
            continue
        pos, euler = _origin(collision)
        attrs = {
            "name": f"{link.attrib['name']}_collision_{collision_index}",
            "pos": _fmt(pos), "euler": _fmt(euler),
            "contype": "1", "conaffinity": "1", "friction": "1 0.005 0.0001",
        }
        box = geometry.find("box")
        cylinder = geometry.find("cylinder")
        sphere = geometry.find("sphere")
        if box is not None:
            size = _floats(box.attrib["size"], 3)
            attrs.update(type="box", size=_fmt([v / 2.0 for v in size]))
        elif cylinder is not None:
            # Isaac Gym's RC-Dog asset is loaded with
            # replace_cylinder_with_capsule=True.  Use the same primitive in
            # MuJoCo; a cylinder has a noticeably different foot/leg contact
            # impulse and can destabilize a policy trained in Isaac Gym.
            attrs.update(type="capsule", size=_fmt([
                float(cylinder.attrib["radius"]), float(cylinder.attrib["length"]) / 2.0
            ]))
        elif sphere is not None:
            attrs.update(type="sphere", size=sphere.attrib["radius"])
        else:
            continue
        ET.SubElement(body, "geom", attrs)
        collision_index += 1


def _build_body(link_name: str, links: dict[str, ET.Element], children: dict[str, list[ET.Element]],
                joints_by_child: dict[str, ET.Element], mesh_names: dict[str, str]) -> ET.Element:
    joint = joints_by_child.get(link_name)
    attrs = {"name": link_name}
    if joint is not None:
        pos, euler = _origin(joint)
        attrs.update(pos=_fmt(pos), euler=_fmt(euler))
    body = ET.Element("body", attrs)
    if joint is not None and joint.attrib["type"] != "fixed":
        axis = joint.find("axis")
        limit = joint.find("limit")
        dynamics = joint.find("dynamics")
        joint_attrs = {"name": joint.attrib["name"], "type": "hinge",
                       "axis": _fmt(_floats(axis.attrib.get("xyz", "0 0 1"), 3)) if axis is not None else "0 0 1"}
        if limit is not None:
            joint_attrs.update(range=f"{limit.attrib['lower']} {limit.attrib['upper']}", limited="true")
        if dynamics is not None and float(dynamics.attrib.get("damping", 0.0)):
            joint_attrs["damping"] = dynamics.attrib["damping"]
        ET.SubElement(body, "joint", joint_attrs)
    _add_inertial(body, links[link_name])
    _add_geoms(body, links[link_name], mesh_names)
    for child_joint in children.get(link_name, []):
        body.append(_build_body(_joint_link(child_joint, "child"), links, children, joints_by_child, mesh_names))
    return body


def convert(urdf_path: Path, xml_path: Path) -> None:
    root = ET.parse(urdf_path).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    joints = root.findall("joint")
    joints_by_child = {_joint_link(joint, "child"): joint for joint in joints}
    children: dict[str, list[ET.Element]] = {}
    for joint in joints:
        children.setdefault(_joint_link(joint, "parent"), []).append(joint)
    child_links = {_joint_link(joint, "child") for joint in joints}
    root_links = [name for name in links if name not in child_links]
    if len(root_links) != 1:
        raise ValueError(f"URDF must have exactly one root link, found {root_links}")
    root_link = root_links[0]

    mujoco = ET.Element("mujoco", {"model": "rc_dog"})
    ET.SubElement(mujoco, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(mujoco, "option", {"timestep": "0.005", "iterations": "10", "integrator": "Euler",
                                       "cone": "elliptic", "solver": "Newton"})
    asset = ET.SubElement(mujoco, "asset")
    # Visible checkerboard material for the MuJoCo floor.
    ET.SubElement(asset, "texture", {"name": "floor_checker", "type": "2d", "builtin": "checker", "width": "32", "height": "32", "rgb1": "0.72 0.72 0.72", "rgb2": "0.38 0.38 0.38"})
    ET.SubElement(asset, "material", {"name": "floor_material", "texture": "floor_checker", "texrepeat": "12 12", "reflectance": "0.1"})
    mesh_names: dict[str, str] = {}
    for link in links.values():
        for visual in link.findall("visual"):
            filename = _mesh_file(visual.find("geometry")) if visual.find("geometry") is not None else None
            if filename is not None:
                mesh_names.setdefault(filename, f"mesh_{_safe_name(Path(filename).stem)}")
    for filename, mesh_name in mesh_names.items():
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": f"../meshes/{filename}"})
    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(worldbody, "geom", {"name": "ground", "type": "plane", "size": "20 20 0.1",
                                       "pos": "0 0 0", "friction": "1 0.005 0.0001", "material": "floor_material"})
    root_body = _build_body(root_link, links, children, joints_by_child, mesh_names)
    ET.SubElement(root_body, "freejoint", {"name": "root"})
    worldbody.append(root_body)

    actuator = ET.SubElement(mujoco, "actuator")
    actuated = [joint for joint in joints if joint.attrib["type"] != "fixed"]
    if len(actuated) != 12:
        raise ValueError(f"expected 12 actuated joints, found {len(actuated)}")
    for joint in actuated:
        limit = joint.find("limit")
        ctrlrange = "-100 100"
        if limit is not None:
            ctrlrange = f"-{limit.attrib['effort']} {limit.attrib['effort']}"
        ET.SubElement(actuator, "motor", {"name": f"motor_{joint.attrib['name']}",
                                            "joint": joint.attrib["name"], "gear": "1",
                                            "ctrllimited": "true", "ctrlrange": ctrlrange})

    _indent(mujoco)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(ET.tostring(mujoco, encoding="unicode") + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=Path("resources/robots/rc_dog/urdf/rc_dog.urdf"))
    parser.add_argument("--xml", type=Path, default=Path("resources/robots/rc_dog/mujoco/rc_dog.xml"))
    args = parser.parse_args()
    convert(args.urdf, args.xml)
    print(f"Wrote {args.xml}")


if __name__ == "__main__":
    main()
