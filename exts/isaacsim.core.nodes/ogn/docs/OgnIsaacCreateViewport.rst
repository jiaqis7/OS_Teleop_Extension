.. _isaacsim_core_nodes_IsaacCreateViewport_2:

.. _isaacsim_core_nodes_IsaacCreateViewport:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Create Viewport
    :keywords: lang-en omnigraph node isaacCore nodes isaac-create-viewport


Isaac Create Viewport
=====================

.. <description>

Isaac Sim node that creates a unique viewport

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.core.nodes<ext_isaacsim_core_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Exec In (*inputs:execIn*)", "``execution``", "Input execution trigger", "None"
    "Name (*inputs:name*)", "``token``", "Name of the viewport window", ""
    "Viewport Id (*inputs:viewportId*)", "``uint``", "If name is empty, ID is used as the name, ID == 0 is the default viewport", "0"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Exec Out (*outputs:execOut*)", "``execution``", "Input execution trigger", "None"
    "Viewport (*outputs:viewport*)", "``token``", "Name of the created viewport", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.core.nodes.IsaacCreateViewport"
    "Version", "2"
    "Extension", "isaacsim.core.nodes"
    "Icon", "ogn/icons/isaacsim.core.nodes.IsaacCreateViewport.svg"
    "Has State?", "False"
    "Implementation Language", "Python"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Isaac Create Viewport"
    "Categories", "isaacCore"
    "Generated Class Name", "OgnIsaacCreateViewportDatabase"
    "Python Module", "isaacsim.core.nodes"

