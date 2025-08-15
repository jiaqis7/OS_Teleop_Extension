.. _isaacsim_core_nodes_IsaacSimulationGate_1:

.. _isaacsim_core_nodes_IsaacSimulationGate:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Simulation Gate
    :keywords: lang-en omnigraph node isaacCore nodes isaac-simulation-gate


Isaac Simulation Gate
=====================

.. <description>

Gate node that only passes through execution if simulation is playing

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
    "Step (*inputs:step*)", "``uint``", "Number of ticks per execution output, default is 1, set to zero or negative numbers to disable execution of connected nodes", "1"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Exec Out (*outputs:execOut*)", "``execution``", "The output execution", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.core.nodes.IsaacSimulationGate"
    "Version", "1"
    "Extension", "isaacsim.core.nodes"
    "Icon", "ogn/icons/isaacsim.core.nodes.IsaacSimulationGate.svg"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Isaac Simulation Gate"
    "Categories", "isaacCore"
    "Generated Class Name", "OgnIsaacSimulationGateDatabase"
    "Python Module", "isaacsim.core.nodes"

