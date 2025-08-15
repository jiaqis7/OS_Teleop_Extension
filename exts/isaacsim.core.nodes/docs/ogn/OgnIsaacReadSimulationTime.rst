.. _isaacsim_core_nodes_IsaacReadSimulationTime_1:

.. _isaacsim_core_nodes_IsaacReadSimulationTime:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Read Simulation Time
    :keywords: lang-en omnigraph node isaacCore nodes isaac-read-simulation-time


Isaac Read Simulation Time
==========================

.. <description>

Holds values related to simulation timestamps

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.core.nodes<ext_isaacsim_core_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Reset On Stop (*inputs:resetOnStop*)", "``bool``", "If True the simulation time will reset when stop is pressed, False means time increases monotonically", "False"
    "Swh Frame Number (*inputs:swhFrameNumber*)", "``int64``", "Optional fabric frame number, leave as zero to get the latest simulation frame time", "0"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Simulation Time (*outputs:simulationTime*)", "``double``", "Current Simulation Time in Seconds", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.core.nodes.IsaacReadSimulationTime"
    "Version", "1"
    "Extension", "isaacsim.core.nodes"
    "Icon", "ogn/icons/isaacsim.core.nodes.IsaacReadSimulationTime.svg"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Isaac Read Simulation Time"
    "Categories", "isaacCore"
    "Generated Class Name", "OgnIsaacReadSimulationTimeDatabase"
    "Python Module", "isaacsim.core.nodes"

