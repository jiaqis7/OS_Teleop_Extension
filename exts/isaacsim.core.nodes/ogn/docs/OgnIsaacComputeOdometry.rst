.. _isaacsim_core_nodes_IsaacComputeOdometry_1:

.. _isaacsim_core_nodes_IsaacComputeOdometry:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Compute Odometry Node
    :keywords: lang-en omnigraph node isaacCore nodes isaac-compute-odometry


Isaac Compute Odometry Node
===========================

.. <description>

Holds values related to odometry, this node is not a replcement for the IMU sensor and the associated Read IMU node

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.core.nodes<ext_isaacsim_core_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Chassis Prim (*inputs:chassisPrim*)", "``target``", "Usd prim reference to the articulation root or rigid body prim", "None"
    "Exec In (*inputs:execIn*)", "``execution``", "The input execution port", "None"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Angular Acceleration (*outputs:angularAcceleration*)", "``vectord[3]``", "Angular acceleration vector in rad/s^2", "[0.0, 0.0, 0.0]"
    "Angular Velocity (*outputs:angularVelocity*)", "``vectord[3]``", "Angular velocity vector in rad/s", "[0.0, 0.0, 0.0]"
    "Exec Out (*outputs:execOut*)", "``execution``", "The output execution port", "None"
    "Linear Acceleration (*outputs:linearAcceleration*)", "``vectord[3]``", "Linear acceleration vector in m/s^2", "[0.0, 0.0, 0.0]"
    "Linear Velocity (*outputs:linearVelocity*)", "``vectord[3]``", "Linear velocity vector in m/s", "[0.0, 0.0, 0.0]"
    "Orientation (*outputs:orientation*)", "``quatd[4]``", "Rotation as a quaternion (IJKR)", "[0.0, 0.0, 0.0, 1.0]"
    "Position (*outputs:position*)", "``vectord[3]``", "Position vector in meters", "[0.0, 0.0, 0.0]"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.core.nodes.IsaacComputeOdometry"
    "Version", "1"
    "Extension", "isaacsim.core.nodes"
    "Icon", "ogn/icons/isaacsim.core.nodes.IsaacComputeOdometry.svg"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Isaac Compute Odometry Node"
    "Categories", "isaacCore"
    "Generated Class Name", "OgnIsaacComputeOdometryDatabase"
    "Python Module", "isaacsim.core.nodes"

