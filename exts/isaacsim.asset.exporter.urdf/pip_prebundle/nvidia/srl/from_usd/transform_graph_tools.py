# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Helper functions for the Transform Graph class."""

# Standard Library
import copy
import shutil
from pathlib import Path
from typing import Dict, List, Optional, cast

# Third Party
import numpy as np

# NVIDIA
import nvidia.srl.usd.prim_helper as prim_helper
from nvidia.srl.basics.types import PathLike
from nvidia.srl.from_usd._from_usd_helper import NodeType, _get_prim_scale
from nvidia.srl.from_usd.transform_graph import TransformEdge, TransformGraph, TransformNode
from nvidia.srl.math.transform import Transform

np.set_printoptions(suppress=True)


def reduce_to_urdf(
    graph: TransformGraph,
    nodes_to_remove: Optional[List[TransformNode]] = None,
    edges_to_remove: Optional[List[TransformEdge]] = None,
    root_node: Optional[TransformNode] = None,
    do_trim_joint_skipping_edges: bool = True,
    do_trim_to_largest_subtree: bool = True,
    do_check_and_fix_geometry_nodes: bool = True,
    do_squash_consecutive_links: bool = True,
    do_align_joint_frames_with_child_frames: bool = True,
    debug_dir: Optional[PathLike] = None,
    delete_debug_dir: bool = False,
) -> None:
    """Reduce the graph to be URDF compliant and optimal.

    This involves attempting make the graph into a single tree by removing loop edges and smaller
    trees, and also remove nodes that create additional fixed links in the URDF file.

    Args:
        graph: The graph object to reduce.
        nodes_to_remove: List of nodes to remove from the `TransformGraph` to break kinematic loops
            and make the graph transformable to something valid to create a URDF.
        edges_to_remove: List of edges to remove from the `TransformGraph` to break kinematic loops
            and make the graph transformable to something valid to create a URDF.
        root_node: The root node that will be set as the root of the kinematic structure of the new
            URDF.  This sets the "robot" element in the new URDF. The root node can either be
            specified with the prim path or with the node name.
        do_trim_joint_skipping_edges: If true, the `trim_joint_skipping_edges` function is called.
        do_trim_to_largest_subtree: If true, the `trim_to_largest_subtree` function is called.
        do_check_and_fix_geometry_nodes: If true, the `check_and_fix_geometry_nodes` function is
            called.
        do_squash_consecutive_links: If true, the `squash_consecutive_links` function is called.
        do_align_joint_frames_with_child_frames: If true, the `align_joint_frames_with_child_frames`
            function is called.
        debug_dir: If set, graphviz images will be generated after each step in the
            given directory.
        delete_debug_dir: If true, the debug directory will be first delete, so all content will be
            freshly created from this run.
    """
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        if delete_debug_dir and debug_dir.exists():
            shutil.rmtree(debug_dir)
        name = f"{graph.name}_0_input"
        graph.render_graphviz(output_dir=debug_dir, name=name)
        graph.render_graphviz(output_dir=debug_dir, name=graph.name)

    # Remove edges
    if edges_to_remove is not None and len(edges_to_remove) > 0:
        for edge in edges_to_remove:
            graph.remove_edge(edge)

        if debug_dir is not None:
            name = f"{graph.name}_1_remove_edges"
            graph.render_graphviz(output_dir=debug_dir, name=name)
            graph.render_graphviz(output_dir=debug_dir, name=graph.name)

    # Remove nodes
    if nodes_to_remove is not None and len(nodes_to_remove) > 0:
        for node in nodes_to_remove:
            graph.remove_node(node)

        if debug_dir is not None:
            name = f"{graph.name}_2_remove_nodes"
            graph.render_graphviz(output_dir=debug_dir, name=name)
            graph.render_graphviz(output_dir=debug_dir, name=graph.name)

    # Remove "from_edges" that connect a link or a geometry node to another link or geometry node if
    # the node connects to a joint
    if do_trim_joint_skipping_edges:
        trim_joint_skipping_edges(graph)

        if debug_dir is not None:
            name = f"{graph.name}_3_trim_joint_skipping_edges"
            graph.render_graphviz(output_dir=debug_dir, name=name)
            graph.render_graphviz(output_dir=debug_dir, name=graph.name)

    rescale_transforms_to_unity(graph)

    # Check transform scaling
    for edge in graph.edges:
        scale = Transform.get_scale(edge.transform)
        if not np.allclose(np.ones(3), scale):
            msg = (
                f"Edge '{edge.name}' transform that does not have unit scaling,"
                f" {Transform.get_scale(edge.transform)}."
            )
            raise RuntimeError(msg)

    # Correct joint edges
    correct_joint_edges(graph)

    if debug_dir is not None:
        name = f"{graph.name}_4_correct_joint_edges"
        graph.render_graphviz(output_dir=debug_dir, name=name)
        graph.render_graphviz(output_dir=debug_dir, name=graph.name)

    # Move true root to the correct position
    move_root(graph)

    if debug_dir is not None:
        name = f"{graph.name}_5_move_root"
        graph.render_graphviz(output_dir=debug_dir, name=name)
        graph.render_graphviz(output_dir=debug_dir, name=graph.name)

    # Trim to largest subtree
    if root_node is None:
        if do_trim_to_largest_subtree:
            # Keep only the nodes that are part of the largest subtree
            trim_to_largest_subtree(graph)
    else:
        # Keep only the nodes that are part of the subtree
        trim_to_subtree_root(graph, root_node)

    if debug_dir is not None:
        name = f"{graph.name}_6_trim_to_subtree"
        graph.render_graphviz(output_dir=debug_dir, name=name)
        graph.render_graphviz(output_dir=debug_dir, name=graph.name)

    # Fix geometry nodes that are not leafs or not connected to links
    if do_check_and_fix_geometry_nodes:
        # check_and_fix_geometry_nodes_old(graph)
        check_and_fix_geometry_nodes(graph)

        if debug_dir is not None:
            name = f"{graph.name}_7_check_and_fix_geometry_nodes"
            graph.render_graphviz(output_dir=debug_dir, name=name)
            graph.render_graphviz(output_dir=debug_dir, name=graph.name)

    # Squash consecutive links
    if do_squash_consecutive_links:
        squash_consecutive_links(graph)

        if debug_dir is not None:
            name = f"{graph.name}_8_squash_consecutive_links"
            graph.render_graphviz(output_dir=debug_dir, name=name)
            graph.render_graphviz(output_dir=debug_dir, name=graph.name)

    # Align joint coordinate frames with child link coordinates frames
    if do_align_joint_frames_with_child_frames:
        align_joint_frames_with_child_frames(graph)

    return


def get_first_link(graph: TransformGraph) -> Optional[TransformNode]:
    """Find the first link in the kinematic chain."""
    joint_nodes = list(filter(lambda node_: node_.type == NodeType.JOINT, graph.nodes))
    if len(joint_nodes) == 0:
        return None

    def recursively_find_first_link(a_joint_node: TransformNode) -> TransformNode:
        a_joint_node_link_prims = prim_helper.get_links_for_joint(a_joint_node.prim)
        if a_joint_node_link_prims[0] is None or a_joint_node_link_prims[1] is None:
            if graph._stage is not None:
                link_node = graph.get_node_from_path(
                    prim_helper.get_path(graph._stage.GetDefaultPrim())
                )
                return link_node
            else:
                msg = "Something is wrong. This should never happen."
                raise RuntimeError(msg)
        try:
            link_node = graph.get_node_from_path(prim_helper.get_path(a_joint_node_link_prims[0]))
        except ValueError:
            link_node = graph.get_node_from_path(prim_helper.get_path(a_joint_node_link_prims[1]))
            return link_node
        joint_neighbors = link_node.get_neighbors_of_type(node_type=NodeType.JOINT)
        if a_joint_node in joint_neighbors:
            joint_neighbors.remove(a_joint_node)
        if len(joint_neighbors) == 0:
            return link_node

        for joint_neighbor in joint_neighbors:
            joint_neighbor_link_prims = prim_helper.get_links_for_joint(joint_neighbor.prim)
            if joint_neighbor_link_prims[1] == link_node.prim:
                break
            else:
                return link_node

        return recursively_find_first_link(joint_neighbor)

    # joint_node = joint_nodes[0]
    joint_node = joint_nodes[-1]
    first_link = recursively_find_first_link(joint_node)
    return first_link


def trim_joint_skipping_edges(graph: TransformGraph) -> None:
    """Conditionally trim edges that connect a link or geometry to a link or geometry.

    Edges only get trimmed if the "from_node" and "to_node" of the edge is not
    connected to a joint and the "from_node" has a joint as a neighbor.

    Loop through all nodes that are not joints. If the node is connected to a joint then from all
    the "from_edges" for that node.

    Args:
        graph: The transform graph to be trimmed.
    """
    min_translation_norm_buffer = 1e-6
    nodes = list(
        filter(
            lambda node_: (not node_.is_leaf)
            and (node_.type == NodeType.LINK or node_.type == NodeType.GEOMETRY),
            graph.nodes,
        )
    )
    edges_to_remove_set = set()
    for node in nodes:
        if any([node_neighbor.type == NodeType.JOINT for node_neighbor in node.neighbors]):
            edges_to_remove_set.update(node.from_edges)

    edges_to_remove = list(edges_to_remove_set)

    first_link = get_first_link(graph)

    # - Group edges by "to node".
    # - If all of the "to edges" are being removed for the "to node" and it is part of the trunk,
    #   then keep one edge.
    #   - Keep the edge that has the transform with the smallest translational norm. The rational is
    #     that this connects the closest frame to the parent frame.

    to_node__edges_to_remove: Dict[TransformNode, List[TransformEdge]] = dict()
    for edge in edges_to_remove:
        if edge.to_node not in to_node__edges_to_remove:
            to_node__edges_to_remove[edge.to_node] = []
        to_node__edges_to_remove[edge.to_node].append(edge)

    for to_node in to_node__edges_to_remove.keys():
        if is_part_of_trunk(to_node) and set(to_node.to_edges) == set(
            to_node__edges_to_remove[to_node]
        ):
            first_link_edge_to_remove = list(
                filter(lambda edge_: edge_.from_node == first_link, to_node.to_edges)
            )
            if len(first_link_edge_to_remove) > 0:
                edge_to_not_remove = first_link_edge_to_remove[0]
            else:
                min_translation_norm = min(
                    [
                        cast(float, np.linalg.norm(Transform.get_translation(edge_.transform)))
                        for edge_ in to_node.to_edges
                    ]
                )
                possible_edges_to_not_remove = list(
                    filter(
                        lambda edge_: np.linalg.norm(Transform.get_translation(edge_.transform))
                        < (min_translation_norm + min_translation_norm_buffer),
                        to_node.to_edges,
                    )
                )

                edge_to_not_remove = min(
                    possible_edges_to_not_remove,
                    key=lambda edge_: len(edge_.from_node.get_neighbors_of_type(NodeType.JOINT)),
                )

            to_node__edges_to_remove[to_node].remove(edge_to_not_remove)
            msg = (
                f"The '{to_node.name}' node is set to have all its to-edges removed in the"
                " `trim_joint_skipping_edges` function. At least one edge needs to be kept. The"
                f" '{edge_to_not_remove.name}' edge is being kept because it has the transform with"
                " the smallest translation norm."
            )
            graph.logger.debug(msg)

        for edge in to_node__edges_to_remove[to_node]:
            graph.remove_edge(edge)

    return


def is_part_of_trunk(node: TransformNode) -> bool:
    """True if the node is part of the tree trunk.

    A "tree trunk" node means it has only one "to edge" and if it has "from edges" there is only
    one that connects to a node that is part of the trunk.
    """
    if node.is_root:
        return True

    def is_part_of_trunk_path(a_node: TransformNode) -> bool:
        if a_node.is_root:
            return len(a_node.to_edges) == 1
        return (
            len(a_node.to_neighbors) == 1
            and len(a_node.to_neighbors[0].to_edges) == 1
            and is_part_of_trunk_path(a_node.to_neighbors[0])
        )

    return is_part_of_trunk_path(node)


def rescale_transforms_to_unity(graph: TransformGraph) -> None:
    """Rescale transforms to unity (i.e. [1, 1, 1])."""

    def recursively_rescale_subgraph(this_node: TransformNode) -> None:
        if this_node.to_global_transform is None:
            msg = "Something is wrong. This should never happen."
            raise RuntimeError(msg)
        assert np.allclose(np.ones(3), Transform.get_scale(this_node.to_global_transform))

        for edge in this_node.to_edges:
            scale = Transform.get_scale(edge.transform)
            scale_transform = Transform.from_scale(scale)
            edge.from_node._scale = this_node._scale @ scale_transform
            if edge.from_node.to_global_transform is None:
                msg = "Something is wrong. This should never happen."
                raise RuntimeError(msg)
            edge.from_node._to_global_transform = Transform.remove_scale(
                edge.from_node.to_global_transform
            )

            if edge.to_node.to_global_transform is None:
                msg = "Something is wrong. This should never happen."
                raise RuntimeError(msg)
            global___to_node = edge.to_node.to_global_transform
            global___from_node = edge.from_node.to_global_transform
            if global___from_node is None:
                msg = "Something is wrong. This should never happen."
                raise RuntimeError(msg)
            to_node___from_node = Transform.inverse(global___to_node) @ global___from_node
            edge._transform = to_node___from_node
            recursively_rescale_subgraph(edge.from_node)
        return

    root_nodes = graph.get_roots()
    for root_node in root_nodes:
        if root_node.to_global_transform is None:
            msg = "Something is wrong here. This should never happen."
            raise RuntimeError(msg)
        root_node._scale = Transform.from_scale(Transform.get_scale(root_node.to_global_transform))
        root_node._to_global_transform = Transform.remove_scale(root_node.to_global_transform)
        recursively_rescale_subgraph(root_node)


def trim_to_largest_subtree(graph: TransformGraph) -> None:
    """Trim to the largest subtree of the graph.

    Note:
        The graph should be partitioned into separate trees, and only trees.

    Args:
        graph: The transform graph to be trimmed.
    """
    largest_subtree_node_cnt = 0
    root_nodes = graph.get_roots()
    if len(root_nodes) == 0:
        msg0 = (
            "Unable to trim to the largest subtree because the transform graph is not partitioned"
            "\ninto separate trees. This usually means there are loops in the graph. Remove the"
            "\nloops by removing nodes and/or edges in the graph."
        )
        msg1 = (
            "Reviewing the Graphviz image of the full transform graph is helpful in deciding what"
            "\nnodes and/or edges to remove. The Graphviz image of the full transform graph can be"
            "\ngenerated with the `usd_to_graphviz` command."
        )
        msg = "\n".join([msg0, msg1])

        raise RuntimeError(msg)
    for root_node in root_nodes:
        subtree_nodes = graph.get_subtree_nodes(root_node)
        if len(subtree_nodes) > largest_subtree_node_cnt:
            largest_subtree_node_cnt = len(subtree_nodes)
            largest_subtree_root_node = root_node

    trim_to_subtree_root(graph, largest_subtree_root_node)


def get_true_root(graph: TransformGraph) -> TransformNode:
    """Try to get the true root node of the graph."""
    possible_roots = graph.get_roots()

    # If there is only one root node return that (zero root nodes should not happen)
    if len(possible_roots) == 0:
        msg = "The graph has no root nodes."
        raise RuntimeError(msg)
    elif len(possible_roots) == 1:
        return possible_roots[0]

    # If one of the possible root nodes is the default prim, then return that

    if graph._stage is None:
        msg = "Something is wrong. This should never happen."
        raise RuntimeError(msg)
    default_prim = graph._stage.GetDefaultPrim()
    possible_root_as_default_prim = list(
        filter(lambda root_node_: root_node_.prim == default_prim, possible_roots)
    )
    if len(possible_root_as_default_prim) == 1:
        return possible_root_as_default_prim[0]
    elif len(possible_root_as_default_prim) > 1:
        msg = (
            "Multiple root nodes are default prims. Something is wrong with the code. This should"
            " never happen"
        )
        raise RuntimeError(msg)

    # Return the possible root node with the shortest prim path string length
    root_with_shortest_prim = min(possible_roots, key=lambda node_: len(str(node_.prim)))
    return root_with_shortest_prim


def correct_joint_edges(graph: TransformGraph) -> None:
    """Flip the direction of joint edges to form a kinematic tree."""
    true_root = get_true_root(graph)

    first_link = true_root
    while not first_link.has_neighbor_of_type(NodeType.JOINT):
        if len(first_link.from_neighbors) >= 1:
            first_link = first_link.from_neighbors[0]
        else:
            return

    # Recursively traverse the joints and connected nodes starting from the root. Flipping joint
    # edges when necessary.
    def recursively_flip_joint_edges(
        this_node: TransformNode, joint_nodes: List[TransformNode]
    ) -> None:
        for joint_node in joint_nodes:
            joint_node_neighbors = [
                neighbor_node
                for neighbor_node in joint_node.neighbors
                if neighbor_node != this_node
            ]
            for edge in joint_node.edges:
                if edge.to_node != this_node:
                    graph.flip_edge(edge)
            for joint_node_neighbor in joint_node_neighbors:
                joint_node_neighbor_joints = list(
                    filter(
                        lambda node_: node_.type == NodeType.JOINT and node_ != joint_node,
                        joint_node_neighbor.neighbors,
                    )
                )
                if len(joint_node_neighbor_joints) > 0:
                    recursively_flip_joint_edges(joint_node_neighbor, joint_node_neighbor_joints)

    joint_nodes = first_link.get_neighbors_of_type(NodeType.JOINT)
    if len(joint_nodes) > 0:
        try:
            recursively_flip_joint_edges(first_link, joint_nodes)
        except RecursionError:
            msg = "Unable to convert this USD to URDF because it has kinematic loops."
            raise RuntimeError(msg)

    return


def move_root(graph: TransformGraph) -> None:
    """Move root links to lowest spot in the joint chain."""
    true_root = min(graph.get_roots(), key=lambda node_: len(str(node_.prim)))
    if len(true_root.from_neighbors) != 1:
        return
    trunk_top_node = true_root
    while True:
        if len(trunk_top_node.from_neighbors) == 1:
            from_neighbor = trunk_top_node.from_neighbors[0]
        else:
            break
        if not (is_part_of_trunk(from_neighbor) and len(from_neighbor.from_neighbors) == 1):
            break
        trunk_top_node = from_neighbor
    assert len(trunk_top_node.to_edges) == 1
    tree_branching_node = trunk_top_node.from_neighbors[0]
    if len(tree_branching_node.from_edges) == 1:
        return
    graph.remove_edge(trunk_top_node.to_edges[0])
    while not tree_branching_node.is_root:
        assert len(trunk_top_node.to_neighbors) == 1
        tree_branching_node = tree_branching_node.to_neighbors[0]

    # TODO (roflaherty): Update `TransformGraph._get_transform_relateive` to do this, then just
    # don't include the `transform` argument in `connect_nodes`.
    to_node = trunk_top_node
    from_node = tree_branching_node
    global___to_node = to_node.to_global_transform
    global___from_node = from_node.to_global_transform
    if global___to_node is None:
        msg = "Something is wrong. This should never happen."
        raise RuntimeError(msg)
    to_node___from_node = Transform.inverse(global___to_node) @ global___from_node
    graph.connect_nodes(to_node, from_node, transform=to_node___from_node)

    return


def trim_to_subtree_root(graph: TransformGraph, root_node: TransformNode) -> None:
    """Trim the graph to a tree with the root starting at the given node."""
    subtree_nodes = graph.get_subtree_nodes(root_node)
    for node in graph.nodes:
        if node not in subtree_nodes:
            graph.remove_node(node)


def squash_consecutive_links(graph: TransformGraph) -> None:
    """Combine consecutive links together.

    The new link name will be the name of the "to neighbor" link. The transforms and all edges are
    updated correctly.
    """
    # `this_node` is the current link node that is being checked to see if it has any link nodes as
    # "from neighbors". If it has a link node as a "from neighbor" then that node is saved in the
    # `from_node` variable. The edge that connects `this_node` (the "to node") to `from_node` (the
    # "from node") is the `this_node_to_edge`. Each of `from_node`'s "from neighbors" are then
    # connected to `this_node` and the `from_node` is removed. The current "from neighbor" of
    # `from_node` is stored in the `from_from_node` variable. The edge that connects `from_node`
    # (the "to node") to `from_from_node` (the "from node") is the `from_node_to_edge`.

    keep_squashing = True

    while keep_squashing:
        keep_squashing = False
        node_removed = False

        # Loop through all nodes starting from the leaves
        for this_node in reversed(graph.sorted_nodes):
            # Skip all non-link nodes
            if this_node.type != NodeType.LINK and this_node.type != NodeType.PHONY:
                continue

            # Loop through this node's "to edges"
            for this_node_to_edge in this_node.to_edges:
                from_node = this_node_to_edge.from_node
                if from_node.type == NodeType.LINK or from_node.type == NodeType.PHONY:
                    # Loop through this node's from node's "to edges"
                    for from_node_to_edge in from_node.to_edges:
                        from_from_node = from_node_to_edge.from_node

                        this_node___from_node = this_node_to_edge.transform
                        from_node___from_from_node = from_node_to_edge.transform
                        this_node___from_from_node = (
                            this_node___from_node @ from_node___from_from_node
                        )

                        graph.connect_nodes(
                            to_node=this_node,
                            from_node=from_from_node,
                            transform=this_node___from_from_node,
                        )

                    graph.remove_node(from_node)

                    node_removed = True
                    keep_squashing = True

                    break

            if node_removed:
                break

    # Check that node and edge transforms are consistent
    if not graph._check_node_and_edge_transforms():
        raise RuntimeError(
            "The node and edge transforms are not consistent. Something is wrong with the code."
        )


def check_and_fix_geometry_nodes(graph: TransformGraph) -> None:
    """Check that the geometry nodes are leaf nodes *and* do not directly connect to joint nodes.

    If they are fix them by creating phony link nodes as necessary.
    """
    for geom_node in filter(lambda node_: node_.type == NodeType.GEOMETRY, graph.nodes):
        if len(geom_node.to_neighbors) > 1:
            msg = (
                "The geometry node '{geom_node.path}' has more than one to-neighbor. This"
                " should never happen."
            )
            raise RuntimeError(msg)

        if not geom_node.is_leaf:
            msg = f"'{geom_node.path}' is not a leaf node. Adding phony link node."
            graph.logger.debug(msg)

            # Add phony link node
            name = geom_node.name + "_link"
            geom_node_link = TransformNode(name, None)
            if geom_node.to_neighbors:
                geom_node_link._to_global_transform = copy.copy(
                    geom_node.to_neighbors[0].to_global_transform
                )
            else:
                geom_node_link._to_global_transform = Transform.identity()

            to_edges = copy.copy(geom_node.to_edges)
            for to_edge in to_edges:
                graph.remove_edge(to_edge)
                to_node = geom_node_link
                from_node = to_edge.from_node
                global___to_node = to_node.to_global_transform
                global___from_node = from_node.to_global_transform
                if global___to_node is None:
                    msg = "Something is wrong. This should never happen."
                    raise RuntimeError(msg)
                to_node___from_node = Transform.inverse(global___to_node) @ global___from_node
                graph.connect_nodes(to_node, from_node, transform=to_node___from_node)

            from_edges = copy.copy(geom_node.from_edges)
            for from_edge in from_edges:
                graph.remove_edge(from_edge)
                to_node = from_edge.to_node
                from_node = geom_node_link
                global___to_node = to_node.to_global_transform
                global___from_node = from_node.to_global_transform
                if global___to_node is None:
                    msg = "Something is wrong. This should never happen."
                    raise RuntimeError(msg)
                to_node___from_node = Transform.inverse(global___to_node) @ global___from_node
                graph.connect_nodes(to_node, from_node, transform=to_node___from_node)

            to_node = geom_node_link
            from_node = geom_node
            global___to_node = to_node.to_global_transform
            global___from_node = from_node.to_global_transform
            if global___to_node is None:
                msg = "Something is wrong. This should never happen."
                raise RuntimeError(msg)
            to_node___from_node = Transform.inverse(global___to_node) @ global___from_node
            graph.connect_nodes(to_node, from_node, transform=to_node___from_node)

        geom_to_neighbor = geom_node.to_neighbors[0]
        if geom_to_neighbor.type != NodeType.LINK and geom_to_neighbor.type != NodeType.PHONY:
            msg = (
                f"The to-neighbor node of '{geom_node.path}', '{geom_to_neighbor.path}', is not a"
                " link node. Adding phony link node."
            )
            graph.logger.debug(msg)

            # Add phony link node
            name = geom_node.name + "_link"
            geom_node_link = TransformNode(name, None)
            if geom_node.to_neighbors:
                geom_node_link._to_global_transform = copy.copy(
                    geom_node.to_neighbors[0].to_global_transform
                )
            else:
                geom_node_link._to_global_transform = Transform.identity()

            geom_node_edge = geom_node.from_edges[0]
            graph.remove_edge(geom_node_edge)

            to_node = geom_node_link
            from_node = geom_node
            global___to_node = to_node.to_global_transform
            global___from_node = from_node.to_global_transform
            if global___to_node is None:
                msg = "Something is wrong. This should never happen."
                raise RuntimeError(msg)
            to_node___from_node = Transform.inverse(global___to_node) @ global___from_node
            graph.connect_nodes(to_node, from_node, transform=to_node___from_node)

            to_node = geom_to_neighbor
            from_node = geom_node_link
            global___to_node = to_node.to_global_transform
            global___from_node = from_node.to_global_transform
            if global___to_node is None:
                msg = "Something is wrong. This should never happen."
                raise RuntimeError(msg)
            to_node___from_node = Transform.inverse(global___to_node) @ global___from_node
            assert np.allclose(Transform.identity(), to_node___from_node)
            graph.connect_nodes(to_node, from_node, transform=to_node___from_node)

    # Check that node and edge transforms are consistent
    if not graph._check_node_and_edge_transforms():
        raise RuntimeError(
            "The node and edge transforms are not consistent. Something is wrong with the code."
        )

    return


def check_and_fix_geometry_nodes_old(graph: TransformGraph) -> None:
    """Check that the geometry nodes are leaf nodes and not directly connected to joint nodes.

    If they are fix them by creating phony link nodes as necessary.
    """
    graph_nodes = graph.nodes
    for geom_node in filter(lambda node_: node_.type == NodeType.GEOMETRY, graph_nodes):
        scale = _get_prim_scale(geom_node.prim)
        if len(geom_node.to_neighbors) != 1:
            msg = (
                "The geometry node '{geom_node.path}' does not have exactly one to-neighbor. This"
                " should never happen."
            )
            raise RuntimeError(msg)

        if not geom_node.is_leaf:
            msg = f"'{geom_node.path}' is not a leaf node. Adding phony link node."
            graph.logger.debug(msg)

            # Add phony link node
            name = geom_node.name + "_link"
            geom_node_link = TransformNode(name, None)

            to_edges = copy.copy(geom_node.to_edges)
            for to_edge in to_edges:
                # TODO (roflaherty): This is not the correct way to apply the geometry
                # scales to the URDF transforms. It is a hack and is only working for the current
                # set of USDs that I have tried.
                #
                # This works for USDs that do not have a rotation on the joint transforms where the
                # scaling is the applied (e.g. cartpole.usd and simple_robot_geom_as_link.usd). I
                # think the better thing to do is refactor the transform graph all together. It
                # should store the transforms wrt to the world then when `UsdToUrdf` in `to_urdf.py`
                # is called it would calculate the local transforms based on the world transform
                # information.  This would allow for the transforms to be set base on where the
                # actual object is in the world (i.e. taking in parent prim scaling information),
                # and the graph nodes can be moved around without needing to update the transform
                # information (i.e. the graph "transformations" that are done in this file). Once
                # the graph is in the right form that is needed for URDF, then the local
                # transformations can be calculated.
                rot = Transform.get_rotation(to_edge.transform)
                trans = Transform.get_translation(to_edge.transform) * scale
                transform = Transform.from_rotmat(rot, trans)
                graph.remove_edge(to_edge)
                graph.connect_nodes(geom_node_link, to_edge.from_node, transform)

            from_edges = copy.copy(geom_node.from_edges)
            for from_edge in from_edges:
                # TODO (roflaherty): Same here... see comment above.
                rot = Transform.get_rotation(to_edge.transform)
                trans = Transform.get_translation(to_edge.transform) * scale
                transform = Transform.from_rotmat(rot, trans)
                graph.remove_edge(from_edge)
                graph.connect_nodes(from_edge.to_node, geom_node_link, transform)

            graph.connect_nodes(geom_node_link, geom_node, Transform.identity())

        geom_to_neighbor = geom_node.to_neighbors[0]
        if geom_to_neighbor.type != NodeType.LINK and geom_to_neighbor.type != NodeType.PHONY:
            msg = (
                f"The to-neighbor node of '{geom_node.path}', '{geom_to_neighbor.path}', is not a"
                " link node. Adding phony link node."
            )
            graph.logger.debug(msg)

            # Add phony link node
            name = geom_node.name + "_link"
            geom_node_link = TransformNode(name, None)
            geom_node_edge = geom_node.from_edges[0]
            transform = geom_node_edge.transform
            graph.remove_edge(geom_node_edge)
            graph.connect_nodes(geom_node_link, geom_node, Transform.identity())
            graph.connect_nodes(geom_to_neighbor, geom_node_link, transform)


def align_joint_frames_with_child_frames(graph: TransformGraph) -> None:
    """Check that the joint child transforms are identity, if not then update them."""
    # Loop through all joint nodes
    for joint_node in filter(lambda node_: node_.type == NodeType.JOINT, graph.nodes):
        # Loop through all joint node "to edges"
        for idx, to_edge in enumerate(joint_node.to_edges):
            if not np.allclose(to_edge._transform, Transform.identity()):
                from_node = to_edge.from_node
                for from_node_to_edge in from_node.to_edges:
                    from_node_to_edge._transform = to_edge._transform @ from_node_to_edge._transform
                to_edge._transform = Transform.identity()
                to_edge.from_node._to_global_transform = to_edge.to_node.to_global_transform

                msg = (
                    f"The to-edge (index: {idx}) for joint '{joint_node.path}' is not set to"
                    " identity. Setting it identity and updating other transforms as necessary."
                )
                # THIS IS AN ERROR BECAUSE I NEED TO CONFIRM IT IS CORRECT
                # raise RuntimeError(msg)
                graph.logger.debug(msg)

    # Check that node and edge transforms are consistent
    if not graph._check_node_and_edge_transforms():
        raise RuntimeError(
            "The node and edge transforms are not consistent. Something is wrong with the code."
        )

    return
