.. _isaacsim_core_nodes_IsaacCreateRenderProduct_2:

.. _isaacsim_core_nodes_IsaacCreateRenderProduct:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Create Render Product
    :keywords: lang-en omnigraph node isaacCore nodes isaac-create-render-product


Isaac Create Render Product
===========================

.. <description>

Isaac Sim node that creates a render product for use with offscreen rendering

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.core.nodes<ext_isaacsim_core_nodes>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Camera Prim (*inputs:cameraPrim*)", "``target``", "Usd prim reference to the camera associated with this render product", "None"
    "Enabled (*inputs:enabled*)", "``bool``", "Set to false to disable downstream execution and RP creation", "True"
    "Exec In (*inputs:execIn*)", "``execution``", "Input execution trigger", "None"
    "Height (*inputs:height*)", "``uint``", "Height of the render product, in pixels", "720"
    "Width (*inputs:width*)", "``uint``", "Width of the render product, in pixels", "1280"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Exec Out (*outputs:execOut*)", "``execution``", "Output execution trigger", "None"
    "Render Product Path (*outputs:renderProductPath*)", "``token``", "Render product path for the created hydra texture", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.core.nodes.IsaacCreateRenderProduct"
    "Version", "2"
    "Extension", "isaacsim.core.nodes"
    "Icon", "ogn/icons/isaacsim.core.nodes.IsaacCreateRenderProduct.svg"
    "Has State?", "False"
    "Implementation Language", "Python"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Isaac Create Render Product"
    "Categories", "isaacCore"
    "Generated Class Name", "OgnIsaacCreateRenderProductDatabase"
    "Python Module", "isaacsim.core.nodes"

