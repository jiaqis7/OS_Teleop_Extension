.. _isaacsim_replicator_domain_randomization_OgnWritePhysicsArticulationView_1:

.. _isaacsim_replicator_domain_randomization_OgnWritePhysicsArticulationView:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Write Physics Attribute using Tensor API
    :keywords: lang-en omnigraph node Replicator domain_randomization ogn-write-physics-articulation-view


Write Physics Attribute using Tensor API
========================================

.. <description>

This node writes physics attributes to TensorAPI views

.. </description>


Installation
------------

To use this node enable :ref:`isaacsim.replicator.domain_randomization<ext_isaacsim_replicator_domain_randomization>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Attribute (*inputs:attribute*)", "``string``", "Name of attribute that is to be written", ""
    "Dist Param 1 (*inputs:dist_param_1*)", "``float[]``", "Distribution parameter 1", "[]"
    "Dist Param 2 (*inputs:dist_param_2*)", "``float[]``", "Distribution parameter 2", "[]"
    "Distribution (*inputs:distribution*)", "``string``", "Type of distribution used to sample values", ""
    "Exec In (*inputs:execIn*)", "``execution``", "exec", "None"
    "Indices (*inputs:indices*)", "``int[]``", "Indices of the environments to assign the physics attribute", "[]"
    "Num Buckets (*inputs:num_buckets*)", "``int``", "Number of buckets to randomize from", "0"
    "On Reset (*inputs:on_reset*)", "``bool``", "indicates whether an on_reset context triggered the execution", "False"
    "Operation (*inputs:operation*)", "``string``", "Type of randomization operation to be applied", ""
    "Prims (*inputs:prims*)", "``string``", "Name of registered view to randomize", ""
    "Values (*inputs:values*)", "``float[]``", "Values to be assigned to the physics attribute", "[]"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Exec Out (*outputs:execOut*)", "``execution``", "exec", "None"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "isaacsim.replicator.domain_randomization.OgnWritePhysicsArticulationView"
    "Version", "1"
    "Extension", "isaacsim.replicator.domain_randomization"
    "Has State?", "False"
    "Implementation Language", "Python"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "tests"
    "uiName", "Write Physics Attribute using Tensor API"
    "Categories", "Replicator"
    "__categoryDescriptions", "Replicator,Write Attribute"
    "Generated Class Name", "OgnWritePhysicsArticulationViewDatabase"
    "Python Module", "isaacsim.replicator.domain_randomization"

