.. _isaacsim_core_nodes_IsaacTimeSplitter_1:

.. _isaacsim_core_nodes_IsaacTimeSplitter:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Time Splitter
    :keywords: lang-en omnigraph node isaacCore nodes isaac-time-splitter


Isaac Time Splitter
===================

.. <description>

Spit time values

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.core.nodes<ext_isaacsim_core_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Time (*inputs:time*)", "``['double', 'float', 'half', 'int', 'int64', 'uint', 'uint64']``", "Time (in seconds)", "None"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Microseconds (*outputs:microseconds*)", "``uint``", "Microseconds [0, 1e6)", "None"
    "Milliseconds (*outputs:milliseconds*)", "``uint``", "Milliseconds [0, 1e3)", "None"
    "Nanoseconds (*outputs:nanoseconds*)", "``uint``", "Nanoseconds [0, 1e9)", "None"
    "Seconds (*outputs:seconds*)", "``int``", "Seconds [INT_MIN, INT_MAX]", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.core.nodes.IsaacTimeSplitter"
    "Version", "1"
    "Extension", "isaacsim.core.nodes"
    "Icon", "ogn/icons/isaacsim.core.nodes.IsaacTimeSplitter.svg"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Isaac Time Splitter"
    "Categories", "isaacCore"
    "Generated Class Name", "OgnIsaacTimeSplitterDatabase"
    "Python Module", "isaacsim.core.nodes"

