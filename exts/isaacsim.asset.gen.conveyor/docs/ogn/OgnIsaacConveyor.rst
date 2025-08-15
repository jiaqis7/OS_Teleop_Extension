.. _isaacsim_asset_gen_conveyor_IsaacConveyor_1:

.. _isaacsim_asset_gen_conveyor_IsaacConveyor:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Conveyor Belt
    :keywords: lang-en omnigraph node isaacSim compute-on-request conveyor isaac-conveyor


Conveyor Belt
=============

.. <description>

Conveyor Belt

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.asset.gen.conveyor<ext_isaacsim_asset_gen_conveyor>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Animate Direction (*inputs:animateDirection*)", "``float[2]``", "relative direction to apply to UV texture", "[1.0, 0.0]"
    "Animate Scale (*inputs:animateScale*)", "``float``", "how fast to scale animate texture", "1.0"
    "Animate Texture (*inputs:animateTexture*)", "``bool``", "If configured, Animates the texture based on the conveyor velocity. may affect performance specially if multiple instances are added to a scene.", "False"
    "Conveyor Prim (*inputs:conveyorPrim*)", "``target``", "the prim reference to the conveyor", "None"
    "Curved (*inputs:curved*)", "``bool``", "If the conveyor belt is curved, check this box to apply angular velocities. The rotation origin will be the rigid body origin.", "False"
    "Delta (*inputs:delta*)", "``float``", "time since last step in seconds", "0.0"
    "Direction (*inputs:direction*)", "``float[3]``", "relative direction to apply velocity", "[1.0, 0.0, 0.0]"
    "Enabled (*inputs:enabled*)", "``bool``", "node does not execute if disabled", "True"
    "On Step (*inputs:onStep*)", "``execution``", "step to animate textures", "None"
    "Velocity (*inputs:velocity*)", "``float``", "the velocity of the conveyor unit", "0.0"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.asset.gen.conveyor.IsaacConveyor"
    "Version", "1"
    "Extension", "isaacsim.asset.gen.conveyor"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Conveyor Belt"
    "Categories", "isaacSim"
    "__categoryDescriptions", "isaacSim,Conveyor belt controller inside Isaac Sim"
    "Generated Class Name", "OgnIsaacConveyorDatabase"
    "Python Module", "isaacsim.asset.gen.conveyor"

