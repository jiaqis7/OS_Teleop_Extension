.. _isaacsim_core_nodes_IsaacGenerateRGBA_1:

.. _isaacsim_core_nodes_IsaacGenerateRGBA:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Generate RGBA
    :keywords: lang-en omnigraph node isaacCore nodes isaac-generate-r-g-b-a


Isaac Generate RGBA
===================

.. <description>

Isaac Sim Node that generates a constant rgba buffer

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.core.nodes<ext_isaacsim_core_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Color (*inputs:color*)", "``colorf[4]``", "Color for output image", "[0.0, 0.0, 0.0, 0.0]"
    "Height (*inputs:height*)", "``uint``", "Height for output image", "100"
    "Width (*inputs:width*)", "``uint``", "Width for output image", "100"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Data (*outputs:data*)", "``uchar[]``", "Buffer rgba array data", "[]"
    "Encoding (*outputs:encoding*)", "``token``", "Encoding as a token", "None"
    "Height (*outputs:height*)", "``uint``", "Height for output image", "None"
    "Width (*outputs:width*)", "``uint``", "Width for output image", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.core.nodes.IsaacGenerateRGBA"
    "Version", "1"
    "Extension", "isaacsim.core.nodes"
    "Icon", "ogn/icons/isaacsim.core.nodes.IsaacGenerateRGBA.svg"
    "Has State?", "False"
    "Implementation Language", "Python"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Isaac Generate RGBA"
    "Categories", "isaacCore"
    "Generated Class Name", "OgnIsaacGenerateRGBADatabase"
    "Python Module", "isaacsim.core.nodes"

