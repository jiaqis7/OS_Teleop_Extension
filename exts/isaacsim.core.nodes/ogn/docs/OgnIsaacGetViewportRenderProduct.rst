.. _isaacsim_core_nodes_IsaacGetViewportRenderProduct_1:

.. _isaacsim_core_nodes_IsaacGetViewportRenderProduct:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Get Viewport Render Product
    :keywords: lang-en omnigraph node isaacCore nodes isaac-get-viewport-render-product


Isaac Get Viewport Render Product
=================================

.. <description>

Isaac Sim node that returns the render product for a given viewport

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
    "Viewport (*inputs:viewport*)", "``token``", "Name of the viewport to get renderproduct for", ""


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

    "Unique ID", "isaacsim.core.nodes.IsaacGetViewportRenderProduct"
    "Version", "1"
    "Extension", "isaacsim.core.nodes"
    "Icon", "ogn/icons/isaacsim.core.nodes.IsaacGetViewportRenderProduct.svg"
    "Has State?", "False"
    "Implementation Language", "Python"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Isaac Get Viewport Render Product"
    "Categories", "isaacCore"
    "Generated Class Name", "OgnIsaacGetViewportRenderProductDatabase"
    "Python Module", "isaacsim.core.nodes"

