.. _isaacsim_core_nodes_OnPhysicsStep_1:

.. _isaacsim_core_nodes_OnPhysicsStep:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: On Physics Step
    :keywords: lang-en omnigraph node event nodes on-physics-step


On Physics Step
===============

.. <description>

Executes an output execution pulse for every physics Simulation Step

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.core.nodes<ext_isaacsim_core_nodes>` in the Extension Manager.


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Simulation Delta Time (*outputs:deltaSimulationTime*)", "``double``", "Simulation Time elapsed since the last update (seconds)", "None"
    "System Delta Time (*outputs:deltaSystemTime*)", "``double``", "System Time elapsed since last update (seconds)", "None"
    "Step (*outputs:step*)", "``execution``", "The execution output", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.core.nodes.OnPhysicsStep"
    "Version", "1"
    "Extension", "isaacsim.core.nodes"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "On Physics Step"
    "Categories", "event"
    "Generated Class Name", "OgnOnPhysicsStepDatabase"
    "Python Module", "isaacsim.core.nodes"

