.. _isaacsim_replicator_domain_randomization_Random3f_1:

.. _isaacsim_replicator_domain_randomization_Random3f:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Random 3f
    :keywords: lang-en omnigraph node domain_randomization random3f


Random 3f
=========

.. <description>

This node outputs the poses of assets with semantic labels

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.replicator.domain_randomization<ext_isaacsim_replicator_domain_randomization>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Maximum (*inputs:maximum*)", "``float[3]``", "minimum range for randomization", "[0.0, 0.0, 0.0]"
    "Minimum (*inputs:minimum*)", "``float[3]``", "minimum range for randomization", "[0.0, 0.0, 0.0]"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Output (*outputs:output*)", "``float[3]``", "Random float 3", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.replicator.domain_randomization.Random3f"
    "Version", "1"
    "Extension", "isaacsim.replicator.domain_randomization"
    "Has State?", "False"
    "Implementation Language", "Python"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Random 3f"
    "Generated Class Name", "OgnRandom3fDatabase"
    "Python Module", "isaacsim.replicator.domain_randomization"

