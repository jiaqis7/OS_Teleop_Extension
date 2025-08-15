.. _isaacsim_core_nodes_IsaacArticulationState_1:

.. _isaacsim_core_nodes_IsaacArticulationState:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Articulation State
    :keywords: lang-en omnigraph node isaacSim nodes isaac-articulation-state


Articulation State
==================

.. <description>

Articulated robot state. The node takes either joint names or joint indices, and outputs the joint positions and velocities, as well as the measured joint efforts, forces, and torques.

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.core.nodes<ext_isaacsim_core_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Exec In (*inputs:execIn*)", "``execution``", "The input execution", "None"
    "Joint Indices (*inputs:jointIndices*)", "``int[]``", "Queried joint indices, use either Joint Names or Joint Indices, if neither is given, default to all joints", "[]"
    "Joint Names (*inputs:jointNames*)", "``token[]``", "Queried joint names, use either Joint Names or Joint Indices, if neither is given, default to all joints", "[]"
    "Robot Path (*inputs:robotPath*)", "``string``", "Path to the robot articulation root. If this is populated, targetPrim is ignored.", ""
    "Target Prim (*inputs:targetPrim*)", "``target``", "The target robot prim with robot articulation root. Ensure robotPath is empty for this to be considered.", "None"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Joint Names (*outputs:jointNames*)", "``token[]``", "Joint names", "None"
    "Joint Positions (*outputs:jointPositions*)", "``double[]``", "Joint positions Units:     linear (m)     angular (rad)", "None"
    "Joint Velocities (*outputs:jointVelocities*)", "``double[]``", "Joint velocities Units:     linear (m/s)     angular (rad/s)", "None"
    "Measured Joint Efforts (*outputs:measuredJointEfforts*)", "``double[]``", "Measured joint efforts Force Units:       linear (kg*m/s^2) i.e. a force     angular (kg*m^2/s^2) i.e. a torque Acceleration Units:     linear (m/s^2), i.e. a linear acceleration     angular (rad/s^2) i.e. an angular acceleration", "None"
    "Measured Joint Forces (*outputs:measuredJointForces*)", "``double[3][]``", "Measured joint reaction forces Force Units:       linear (kg*m/s^2) i.e. a force     angular (kg*m^2/s^2) i.e. a torque Acceleration Units:     linear (m/s^2), i.e. a linear acceleration     angular (rad/s^2) i.e. an angular acceleration", "None"
    "Measured Joint Torques (*outputs:measuredJointTorques*)", "``double[3][]``", "Measured joint reaction torques Force Units:       linear (kg*m/s^2) i.e. a force     angular (kg*m^2/s^2) i.e. a torque Acceleration Units:     linear (m/s^2), i.e. a linear acceleration     angular (rad/s^2) i.e. an angular acceleration", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.core.nodes.IsaacArticulationState"
    "Version", "1"
    "Extension", "isaacsim.core.nodes"
    "Has State?", "False"
    "Implementation Language", "Python"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Articulation State"
    "Categories", "isaacSim"
    "__categoryDescriptions", "isaacSim,robot state inside Isaac Sim"
    "Generated Class Name", "OgnIsaacArticulationStateDatabase"
    "Python Module", "isaacsim.core.nodes"

