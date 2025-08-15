# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""USD to URDF CLI."""

# Standard Library
import argparse
import sys
from pathlib import Path

# NVIDIA
import nvidia.srl.from_usd
import nvidia.srl.tools.logger as logger
from nvidia.srl.from_usd.to_urdf import UsdToUrdf


def main() -> None:
    """Run usd_to_urdf CLI."""
    parser = argparse.ArgumentParser("USD to URDF.")

    # Check if '--version' is passed as the only argument
    if "--version" in sys.argv:
        print(nvidia.srl.from_usd.__version__)
        return

    parser.add_argument("input", type=str, help="The input file path.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=(
            "The output URDF file path or output directory. If set to a directory, the URDF file"
            " will be saved in the given directory with the name set to the same name as the input"
            " USD file but with the .urdf extension. If not provided the output will be printed to"
            " the terminal."
        ),
    )
    parser.add_argument(
        "-r",
        "--root",
        type=str,
        default=None,
        help=(
            "The root of the robot's kinematic tree. Can either be specified with the prim path"
            " or with the TransformGraph node name. Defaults to the path of the stage's default"
            " prim."
        ),
    )
    parser.add_argument(
        "-m",
        "--mesh-dir",
        type=str,
        default=None,
        help=(
            "Set the path to the directory where the mesh files will be stored. Defaults to"
            " `meshes` directory in the same directory as the URDF file."
        ),
    )
    parser.add_argument(
        "-p",
        "--mesh-path-prefix",
        type=str,
        default="",
        help=(
            "Set the prefix to use for the URDF mesh filename. For example, to use an absolute path"
            " set this to '$(pwd)/'. Or to use a URI with the 'file' scheme, then set this to"
            " 'file://'."
        ),
    )
    parser.add_argument(
        "--use-uri-file-prefix",
        action="store_true",
        help=(
            "If set, ignore the `--mesh-path-prefix` and set it to use a URI with the 'file' scheme"
            " with the absolute path set to the output directory (i.e. 'file://<absolute path to"
            " output directory>/')."
        ),
    )
    parser.add_argument(
        "--visualize-collision-meshes",
        action="store_true",
        help=(
            "If set, the collision meshes will be"
            " included in the set of visual geometries in the URDF."
        ),
    )
    parser.add_argument(
        "--kinematics-only",
        action="store_true",
        help="If set, the URDF will not contain any visual or collision mesh information.",
    )
    parser.add_argument(
        "--save-graphviz",
        action="store_true",
        help=(
            "If set, a Graphviz visualization of the transform graph is saved to file. Both the"
            " rendered PNG file and the Graphviz dot text file are generated and saved to the same"
            " directory the `meshes` directory is located."
        ),
    )
    parser.add_argument(
        "--remove-nodes",
        nargs="+",
        help=(
            "Names of nodes that should be removed from the TransformGraph so that a valid URDF"
            " TransformGraph can be made."
        ),
    )
    parser.add_argument(
        "--remove-edges",
        nargs="+",
        help=(
            "Names of edges that should be removed from the TransformGraph so that a valid URDF"
            " TransformGraph can be made."
        ),
    )
    parser.add_argument(
        "--parent-link-is-body-1",
        nargs="+",
        help=(
            "Names of joint nodes where the parent link is assumed to be the body 1 target prim,"
            " instead of the default body 0 target prim. Note, when only one body target is set,"
            " then the parent link is assumed to be the default prim, and the child link is the"
            " prim of the set body target."
        ),
    )
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        choices=["critical", "error", "warning", "info", "debug"],
        default="info",
        help="Set the log level.",
    )
    parser.add_argument("--version", action="store_true", help="Print the current version.")

    args = parser.parse_args()

    log_level = logger.level_from_name(args.log_level.upper())

    usd_to_urdf = UsdToUrdf.init_from_file(
        args.input,
        root=args.root,
        node_names_to_remove=args.remove_nodes,
        edge_names_to_remove=args.remove_edges,
        parent_link_is_body_1=args.parent_link_is_body_1,
        log_level=log_level,
    )
    if args.output is None:
        output_str = usd_to_urdf.to_str(
            mesh_dir=args.mesh_dir,
            mesh_path_prefix=args.mesh_path_prefix,
            use_uri_file_prefix=args.use_uri_file_prefix,
            visualize_collision_meshes=args.visualize_collision_meshes,
            kinematics_only=args.kinematics_only,
        )
        print(output_str)
    else:
        usd_to_urdf.save_to_file(
            urdf_output_path=args.output,
            mesh_dir=args.mesh_dir,
            mesh_path_prefix=args.mesh_path_prefix,
            use_uri_file_prefix=args.use_uri_file_prefix,
            visualize_collision_meshes=args.visualize_collision_meshes,
            kinematics_only=args.kinematics_only,
        )
    if args.save_graphviz:
        if args.output is None:
            output_dir = None
        else:
            output_dir = Path(args.output).parent
        usd_to_urdf.save_graphviz(output_dir=output_dir)


if __name__ == "__main__":
    main()
