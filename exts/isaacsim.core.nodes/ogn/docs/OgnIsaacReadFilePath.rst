.. _isaacsim_core_nodes_IsaacReadFilePath_1:

.. _isaacsim_core_nodes_IsaacReadFilePath:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Read File Path
    :keywords: lang-en omnigraph node isaacCore nodes isaac-read-file-path


Isaac Read File Path
====================

.. <description>

Loads contents of file when given path, if file exists

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.core.nodes<ext_isaacsim_core_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Input Path (*inputs:path*)", "``path``", "Input path to file", ""


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Output File Contents (*outputs:fileContents*)", "``['string', 'token']``", "Output contents of file at path, returns empty string if file is not found", ""


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.core.nodes.IsaacReadFilePath"
    "Version", "1"
    "Extension", "isaacsim.core.nodes"
    "Icon", "ogn/icons/isaacsim.core.nodes.IsaacReadFilePath.svg"
    "Has State?", "False"
    "Implementation Language", "Python"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Isaac Read File Path"
    "Categories", "isaacCore"
    "Generated Class Name", "OgnIsaacReadFilePathDatabase"
    "Python Module", "isaacsim.core.nodes"

