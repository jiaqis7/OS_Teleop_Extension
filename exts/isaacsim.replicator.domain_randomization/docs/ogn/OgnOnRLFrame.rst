.. _isaacsim_replicator_domain_randomization_OgnOnRLFrame_1:

.. _isaacsim_replicator_domain_randomization_OgnOnRLFrame:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: On Frame
    :keywords: lang-en omnigraph node Replicator compute-on-request domain_randomization ogn-on-r-l-frame


On Frame
========

.. <description>

Triggered every frame in an Rl setting

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.replicator.domain_randomization<ext_isaacsim_replicator_domain_randomization>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Num Envs (*inputs:num_envs*)", "``int``", "number of RL environments", "0"
    "Run (*inputs:run*)", "``bool``", "Run", "False"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Exec Out (*outputs:execOut*)", "``execution``", "Output Execution", "None"
    "Frame Num (*outputs:frameNum*)", "``int[]``", "frame number for every environment", "None"
    "Reset Inds (*outputs:resetInds*)", "``int[]``", "indices of environments to be reset", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.replicator.domain_randomization.OgnOnRLFrame"
    "Version", "1"
    "Extension", "isaacsim.replicator.domain_randomization"
    "Has State?", "True"
    "Implementation Language", "Python"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "On Frame"
    "Categories", "Replicator"
    "__categoryDescriptions", "Replicator,On Frame"
    "Generated Class Name", "OgnOnRLFrameDatabase"
    "Python Module", "isaacsim.replicator.domain_randomization"

