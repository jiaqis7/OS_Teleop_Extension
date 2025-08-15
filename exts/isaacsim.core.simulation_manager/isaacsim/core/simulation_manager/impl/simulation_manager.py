import weakref
from collections import OrderedDict

import carb
import isaacsim.core.utils.numpy as np_utils
import isaacsim.core.utils.torch as torch_utils
import isaacsim.core.utils.warp as warp_utils
import omni.kit
import omni.physx
import omni.timeline
import omni.usd
from pxr import PhysxSchema

from .isaac_events import IsaacEvents


class SimulationManager:
    """This class provide functions that take care of many time-related events such as
    warm starting simulation in order for the physics data to be retrievable.
    Adding/ removing callback functions that gets triggered with certain events such as a physics step,
    on post reset, on physics ready..etc."""

    _warmup_needed = True
    _warm_start_callback = None
    _on_stop_callback = None
    _timeline = omni.timeline.get_timeline_interface()
    _message_bus = omni.kit.app.get_app().get_message_bus_event_stream()
    _physx_sim_interface = omni.physx.get_physx_simulation_interface()
    _physx_interface = omni.physx.acquire_physx_interface()
    _physics_sim_view = None
    _post_warm_start_callback = None
    _stage_open_callback = None
    _backend = "numpy"
    _carb_settings = carb.settings.get_settings()
    _physics_scene_apis = OrderedDict()
    _callbacks = dict()
    _simulation_manager_interface = None
    _assets_loaded = True
    _assets_loading_callback = None

    @classmethod
    def _initialize(cls) -> None:
        SimulationManager._warm_start_callback = (
            SimulationManager._timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.PLAY), SimulationManager._warm_start
            )
        )
        SimulationManager._on_stop_callback = (
            SimulationManager._timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.STOP), SimulationManager._on_stop
            )
        )
        SimulationManager._post_warm_start_callback = SimulationManager._message_bus.create_subscription_to_pop_by_type(
            IsaacEvents.PHYSICS_WARMUP.value, SimulationManager._create_simulation_view
        )
        SimulationManager._stage_open_callback = (
            omni.usd.get_context()
            .get_stage_event_stream()
            .create_subscription_to_pop_by_type(int(omni.usd.StageEventType.OPENED), SimulationManager._post_stage_open)
        )
        SimulationManager._track_physics_scenes()

    @classmethod
    def _clear(cls) -> None:
        SimulationManager._warm_start_callback = None
        SimulationManager._on_stop_callback = None
        SimulationManager._physics_sim_view = None
        SimulationManager._post_warm_start_callback = None
        SimulationManager._stage_open_callback = None
        SimulationManager._assets_loading_callback = None
        SimulationManager._simulation_manager_interface.reset()
        SimulationManager._physics_scene_apis.clear()
        SimulationManager._callbacks.clear()

    def _post_stage_open(event) -> None:
        SimulationManager._simulation_manager_interface.reset()
        SimulationManager._physics_scene_apis.clear()
        SimulationManager._callbacks.clear()
        SimulationManager._track_physics_scenes()
        SimulationManager._assets_loaded = True
        SimulationManager._assets_loading_callback = None

        def on_stage_event(event: omni.usd.StageEventType):
            if event.type == int(omni.usd.StageEventType.ASSETS_LOADING):
                SimulationManager._assets_loaded = False
            elif event.type == int(omni.usd.StageEventType.ASSETS_LOADED):
                SimulationManager._assets_loaded = True

        SimulationManager._assets_loading_callback = (
            omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(on_stage_event)
        )

    def _track_physics_scenes() -> None:
        def add_physics_scenes(physics_scene_prim_path):
            prim = omni.usd.get_context().get_stage().GetPrimAtPath(physics_scene_prim_path)
            if prim.GetTypeName() == "PhysicsScene":
                SimulationManager._physics_scene_apis[physics_scene_prim_path] = PhysxSchema.PhysxSceneAPI.Apply(prim)

        def remove_physics_scenes(physics_scene_prim_path):
            # TODO: match physics scene prim path
            if physics_scene_prim_path in SimulationManager._physics_scene_apis:
                del SimulationManager._physics_scene_apis[physics_scene_prim_path]

        SimulationManager._simulation_manager_interface.register_physics_scene_addition_callback(add_physics_scenes)
        SimulationManager._simulation_manager_interface.register_deletion_callback(remove_physics_scenes)

    def _warm_start(event) -> None:
        if SimulationManager._warmup_needed:
            SimulationManager._physx_interface.force_load_physics_from_usd()
            SimulationManager._physx_interface.start_simulation()
            SimulationManager._physx_interface.update_simulation(SimulationManager.get_physics_dt(), 0.0)
            SimulationManager._physx_sim_interface.fetch_results()
            SimulationManager._message_bus.dispatch(IsaacEvents.PHYSICS_WARMUP.value, payload={})
            SimulationManager._warmup_needed = False

    def _on_stop(event) -> None:
        SimulationManager._warmup_needed = True
        if SimulationManager._physics_sim_view:
            SimulationManager._physics_sim_view.invalidate()
            SimulationManager._physics_sim_view = None

    def _create_simulation_view(event) -> None:
        if "cuda" in SimulationManager.get_physics_sim_device() and SimulationManager._backend == "numpy":
            SimulationManager._backend = "torch"
            carb.log_warn("changing backend from numpy to torch since numpy backend cannot be used with GPU piplines")
        SimulationManager._physics_sim_view = omni.physics.tensors.create_simulation_view(SimulationManager._backend)
        SimulationManager._physics_sim_view.set_subspace_roots("/")
        SimulationManager._physx_interface.update_simulation(SimulationManager.get_physics_dt(), 0.0)
        SimulationManager._message_bus.dispatch(IsaacEvents.SIMULATION_VIEW_CREATED.value, payload={})
        SimulationManager._message_bus.dispatch(IsaacEvents.PHYSICS_READY.value, payload={})

    @classmethod
    def _get_backend_utils(cls) -> str:
        if SimulationManager._backend == "numpy":
            return np_utils
        elif SimulationManager._backend == "torch":
            return torch_utils
        elif SimulationManager._backend == "warp":
            return warp_utils
        else:
            raise Exception(
                f"Provided backend is not supported: {SimulationManager.get_backend()}. Supported: torch, numpy, warp."
            )

    @classmethod
    def set_backend(cls, val: str) -> None:
        SimulationManager._backend = val

    @classmethod
    def get_backend(cls) -> str:
        return SimulationManager._backend

    @classmethod
    def get_physics_sim_view(cls):
        return SimulationManager._physics_sim_view

    @classmethod
    def set_physics_sim_device(cls, val) -> None:
        if "cuda" in val:
            parsed_device = val.split(":")
            if len(parsed_device) == 1:
                device_id = SimulationManager._carb_settings.get_as_int("/physics/cudaDevice")
                if device_id < 0:
                    SimulationManager._carb_settings.set_int("/physics/cudaDevice", 0)
                    device_id = 0
            else:
                SimulationManager._carb_settings.set_int("/physics/cudaDevice", int(parsed_device[1]))
            SimulationManager._carb_settings.set_bool("/physics/suppressReadback", True)
            SimulationManager.set_broadphase_type("GPU")
            SimulationManager.enable_gpu_dynamics(flag=True)
            SimulationManager.enable_fabric(enable=True)
        elif "cpu" == val.lower():
            SimulationManager._carb_settings.set_bool("/physics/suppressReadback", False)
            # SimulationManager._carb_settings.set_int("/physics/cudaDevice", -1)
            SimulationManager.set_broadphase_type("MBP")
            SimulationManager.enable_gpu_dynamics(flag=False)
        else:
            raise Exception("Device {} is not supported.".format(val))

    @classmethod
    def get_physics_sim_device(cls) -> str:
        supress_readback = SimulationManager._carb_settings.get_as_bool("/physics/suppressReadback")
        if (not SimulationManager._physics_scene_apis and supress_readback) or (
            supress_readback
            and SimulationManager.get_broadphase_type() == "GPU"
            and SimulationManager.is_gpu_dynamics_enabled()
        ):
            device_id = SimulationManager._carb_settings.get_as_int("/physics/cudaDevice")
            if device_id < 0:
                SimulationManager._carb_settings.set_int("/physics/cudaDevice", 0)
                device_id = 0
            return f"cuda:{device_id}"
        else:
            return "cpu"

    @classmethod
    def get_physics_dt(cls, physics_scene: str = None) -> str:
        """
         Returns the current physics dt.

        Args:
            physics_scene (str, optional): physics scene prim path.

        Raises:
            Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.

        Returns:
            float: physics dt.
        """
        if physics_scene is None:
            if len(SimulationManager._physics_scene_apis) > 0:
                physx_scene_api = list(SimulationManager._physics_scene_apis.values())[-1]
                physics_hz = physx_scene_api.GetTimeStepsPerSecondAttr().Get()
                if physics_hz == 0:
                    return 0.0
                else:
                    return 1.0 / physics_hz
            else:
                return 1.0 / 60.0
        else:
            if physics_scene in SimulationManager._physics_scene_apis:
                physx_scene_api = SimulationManager._physics_scene_apis[physics_scene]
                physics_hz = physx_scene_api.GetTimeStepsPerSecondAttr().Get()
                if physics_hz == 0:
                    return 0.0
                else:
                    return 1.0 / physics_hz
            else:
                raise Exception("physics scene specified {} doesn't exist".format(physics_scene))

    @classmethod
    def get_broadphase_type(cls, physics_scene: str = None) -> str:
        """Gets current broadcast phase algorithm type.

        Args:
            physics_scene (str, optional): physics scene prim path.

        Raises:
            Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.

        Returns:
            str: Broadcast phase algorithm used.
        """
        if physics_scene is None:
            if len(SimulationManager._physics_scene_apis) > 0:
                physx_scene_api = list(SimulationManager._physics_scene_apis.values())[-1]
                return physx_scene_api.GetBroadphaseTypeAttr().Get()
            else:
                return "MBP"
        else:
            if physics_scene in SimulationManager._physics_scene_apis:
                physx_scene_api = SimulationManager._physics_scene_apis[physics_scene]
                return physx_scene_api.GetBroadphaseTypeAttr().Get()
            else:
                raise Exception("physics scene specified {} doesn't exist".format(physics_scene))

    @classmethod
    def set_broadphase_type(cls, val: str, physics_scene: str = None) -> None:
        """Broadcast phase algorithm used in simulation.

        Args:
            val (str): type of broadcasting to be used, can be "MBP".
            physics_scene (str, optional): physics scene prim path.

        Raises:
            Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.
        """
        if physics_scene is None:
            for path, physx_scene_api in SimulationManager._physics_scene_apis.items():
                if physx_scene_api.GetBroadphaseTypeAttr().Get() is None:
                    physx_scene_api.CreateBroadphaseTypeAttr(val)
                else:
                    physx_scene_api.GetBroadphaseTypeAttr().Set(val)
        else:
            if physics_scene in SimulationManager._physics_scene_apis:
                physx_scene_api = SimulationManager._physics_scene_apis[physics_scene]
                if physx_scene_api.GetBroadphaseTypeAttr().Get() is None:
                    physx_scene_api.CreateBroadphaseTypeAttr(val)
                else:
                    physx_scene_api.GetBroadphaseTypeAttr().Set(val)
            else:
                raise Exception("physics scene specified {} doesn't exist".format(physics_scene))

    @classmethod
    def enable_gpu_dynamics(cls, flag: bool, physics_scene: str = None) -> None:
        """Enables gpu dynamics pipeline, required for deformables for instance.

        Args:
            flag (bool): enables or disables gpu dynamics on the PhysicsScene.
            physics_scene (str, optional): physics scene prim path.

        Raises:
            Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.
        """
        if physics_scene is None:
            for path, physx_scene_api in SimulationManager._physics_scene_apis.items():
                if physx_scene_api.GetEnableGPUDynamicsAttr().Get() is None:
                    physx_scene_api.CreateEnableGPUDynamicsAttr(flag)
                else:
                    physx_scene_api.GetEnableGPUDynamicsAttr().Set(flag)
        else:
            if physics_scene in SimulationManager._physics_scene_apis:
                physx_scene_api = SimulationManager._physics_scene_apis[physics_scene]
                if physx_scene_api.GetEnableGPUDynamicsAttr().Get() is None:
                    physx_scene_api.CreateEnableGPUDynamicsAttr(flag)
                else:
                    physx_scene_api.GetEnableGPUDynamicsAttr().Set(flag)
            else:
                raise Exception("physics scene specified {} doesn't exist".format(physics_scene))

    @classmethod
    def is_gpu_dynamics_enabled(cls, physics_scene: str = None) -> bool:
        """Checks if Gpu Dynamics is enabled.

        Args:
            physics_scene (str, optional): physics scene prim path.

        Raises:
            Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.

        Returns:
            bool: True if Gpu Dynamics is enabled, otherwise False.
        """
        if physics_scene is None:
            if len(SimulationManager._physics_scene_apis) > 0:
                physx_scene_api = list(SimulationManager._physics_scene_apis.values())[-1]
                return physx_scene_api.GetEnableGPUDynamicsAttr().Get()
            else:
                return False
        else:
            if physics_scene in SimulationManager._physics_scene_apis:
                physx_scene_api = SimulationManager._physics_scene_apis[physics_scene]
                return physx_scene_api.GetEnableGPUDynamicsAttr().Get()
            else:
                raise Exception("physics scene specified {} doesn't exist".format(physics_scene))

    @classmethod
    def enable_fabric(cls, enable):
        manager = omni.kit.app.get_app().get_extension_manager()
        fabric_was_enabled = manager.is_extension_enabled("omni.physx.fabric")
        if not fabric_was_enabled and enable:
            manager.set_extension_enabled_immediate("omni.physx.fabric", True)
        elif fabric_was_enabled and not enable:
            manager.set_extension_enabled_immediate("omni.physx.fabric", False)
        SimulationManager._carb_settings.set_bool("/physics/updateToUsd", not enable)
        SimulationManager._carb_settings.set_bool("/physics/updateParticlesToUsd", not enable)
        SimulationManager._carb_settings.set_bool("/physics/updateVelocitiesToUsd", not enable)
        SimulationManager._carb_settings.set_bool("/physics/updateForceSensorsToUsd", not enable)
        SimulationManager._carb_settings.set_bool("/physics/outputVelocitiesLocalSpace", not enable)

    @classmethod
    def is_fabric_enabled(cls, enable):
        return omni.kit.app.get_app().get_extension_manager().is_extension_enabled("omni.physx.fabric")

    @classmethod
    def set_solver_type(cls, solver_type: str, physics_scene: str = None) -> None:
        """solver used for simulation.

        Args:
            solver_type (str): can be "TGS" or "PGS".
            physics_scene (str, optional): physics scene prim path.

        Raises:
            Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.
        """
        if solver_type not in ["TGS", "PGS"]:
            raise Exception("Solver type {} is not supported".format(solver_type))

        if physics_scene is None:
            if len(SimulationManager._physics_scene_apis) > 0:
                physx_scene_api = list(SimulationManager._physics_scene_apis.values())[-1]
                if physx_scene_api.GetSolverTypeAttr().Get() is None:
                    physx_scene_api.CreateSolverTypeAttr(solver_type)
                else:
                    physx_scene_api.GetSolverTypeAttr().Set(solver_type)
            else:
                raise Exception("No physics scenes in the stage")
        else:
            if physics_scene in SimulationManager._physics_scene_apis:
                physx_scene_api = SimulationManager._physics_scene_apis[physics_scene]
                if physx_scene_api.GetSolverTypeAttr().Get() is None:
                    physx_scene_api.CreateSolverTypeAttr(solver_type)
                else:
                    physx_scene_api.GetSolverTypeAttr().Set(solver_type)
            else:
                raise Exception("physics scene specified {} doesn't exist".format(physics_scene))

    @classmethod
    def get_solver_type(cls, physics_scene: str = None) -> str:
        """Gets current solver type.

        Args:
            physics_scene (str, optional): physics scene prim path.

        Raises:
            Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.

        Returns:
            str: solver used for simulation.
        """
        if physics_scene is None:
            if len(SimulationManager._physics_scene_apis) > 0:
                physx_scene_api = list(SimulationManager._physics_scene_apis.values())[-1]
                return physx_scene_api.GetSolverTypeAttr().Get()
            else:
                raise Exception("No physics scenes in the stage")
        else:
            if physics_scene in SimulationManager._physics_scene_apis:
                physx_scene_api = SimulationManager._physics_scene_apis[physics_scene]
                return physx_scene_api.GetSolverTypeAttr().Get()
            else:
                raise Exception("physics scene specified {} doesn't exist".format(physics_scene))

    @classmethod
    def enable_stablization(cls, flag: bool, physics_scene: str = None) -> None:
        """Enables additional stabilization pass in the solver.

        Args:
            flag (bool): enables or disables stabilization on the PhysicsScene
            physics_scene (str, optional): physics scene prim path.

        Raises:
            Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.
        """
        if physics_scene is None:
            for path, physx_scene_api in SimulationManager._physics_scene_apis.items():
                if physx_scene_api.GetEnableStabilizationAttr().Get() is None:
                    physx_scene_api.CreateEnableStabilizationAttr(flag)
                else:
                    physx_scene_api.GetEnableStabilizationAttr().Set(flag)
        else:
            if physics_scene in SimulationManager._physics_scene_apis:
                physx_scene_api = SimulationManager._physics_scene_apis[physics_scene]
                if physx_scene_api.GetEnableStabilizationAttr().Get() is None:
                    physx_scene_api.CreateEnableStabilizationAttr(flag)
                else:
                    physx_scene_api.GetEnableStabilizationAttr().Set(flag)
            else:
                raise Exception("physics scene specified {} doesn't exist".format(physics_scene))

    @classmethod
    def is_stablization_enabled(cls, physics_scene: str = None) -> bool:
        """Checks if stabilization is enabled.

        Args:
            physics_scene (str, optional): physics scene prim path.

        Raises:
            Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.

        Returns:
            bool: True if stabilization is enabled, otherwise False.
        """
        if physics_scene is None:
            if len(SimulationManager._physics_scene_apis) > 0:
                physx_scene_api = list(SimulationManager._physics_scene_apis.values())[-1]
                return physx_scene_api.GetEnableStabilizationAttr().Get()
            else:
                return False
        else:
            if physics_scene in SimulationManager._physics_scene_apis:
                physx_scene_api = SimulationManager._physics_scene_apis[physics_scene]
                return physx_scene_api.GetEnableStabilizationAttr().Get()
            else:
                raise Exception("physics scene specified {} doesn't exist".format(physics_scene))

    @classmethod
    def register_callback(cls, callback: callable, event):
        proxy_needed = False
        if hasattr(callback, "__self__"):
            proxy_needed = True
            callback_name = callback.__name__
        callback_id = SimulationManager._simulation_manager_interface.get_callback_iter()
        if event in [
            IsaacEvents.PHYSICS_WARMUP,
            IsaacEvents.PHYSICS_READY,
            IsaacEvents.POST_RESET,
            IsaacEvents.SIMULATION_VIEW_CREATED,
        ]:
            if proxy_needed:
                SimulationManager._callbacks[
                    callback_id
                ] = SimulationManager._message_bus.create_subscription_to_pop_by_type(
                    event.value, lambda event, obj=weakref.proxy(callback.__self__): getattr(obj, callback_name)(event)
                )
            else:
                SimulationManager._callbacks[
                    callback_id
                ] = SimulationManager._message_bus.create_subscription_to_pop_by_type(event.value, callback)
        elif event == IsaacEvents.PRIM_DELETION:
            if proxy_needed:
                SimulationManager._simulation_manager_interface.register_deletion_callback(
                    lambda event, obj=weakref.proxy(callback.__self__): getattr(obj, callback_name)(event)
                )
            else:
                SimulationManager._simulation_manager_interface.register_deletion_callback(callback)
        elif event == IsaacEvents.PHYSICS_STEP:
            if proxy_needed:
                SimulationManager._callbacks[
                    callback_id
                ] = omni.physx.acquire_physx_interface().subscribe_physics_step_events(
                    lambda step_dt, obj=weakref.proxy(callback.__self__): getattr(obj, callback_name)(step_dt)
                )
            else:
                SimulationManager._callbacks[
                    callback_id
                ] = omni.physx.acquire_physx_interface().subscribe_physics_step_events(callback)
        elif event == IsaacEvents.TIMELINE_STOP:
            if proxy_needed:
                SimulationManager._callbacks[
                    callback_id
                ] = SimulationManager._timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
                    int(omni.timeline.TimelineEventType.STOP),
                    lambda event, obj=weakref.proxy(callback.__self__): getattr(obj, callback_name)(event),
                )
            else:
                SimulationManager._callbacks[
                    callback_id
                ] = SimulationManager._timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
                    int(omni.timeline.TimelineEventType.STOP), callback
                )
        else:
            raise Exception("{} event doesn't exist for callback registering".format(event))
        SimulationManager._simulation_manager_interface.set_callback_iter(callback_id + 1)
        return callback_id

    @classmethod
    def deregister_callback(cls, callback_id):
        if callback_id in SimulationManager._callbacks:
            del SimulationManager._callbacks[callback_id]
        elif SimulationManager._simulation_manager_interface.deregister_callback(callback_id):
            return
        else:
            raise Exception("callback with id {} doesn't exist to be deregistered".format(callback_id))

    @classmethod
    def enable_usd_notice_handler(cls, flag):
        SimulationManager._simulation_manager_interface.enable_usd_notice_handler(flag)
        return

    @classmethod
    def enable_fabric_usd_notice_handler(cls, stage_id, flag):
        SimulationManager._simulation_manager_interface.enable_fabric_usd_notice_handler(stage_id, flag)
        return

    @classmethod
    def is_fabric_usd_notice_handler_enabled(cls, stage_id):
        return SimulationManager._simulation_manager_interface.is_fabric_usd_notice_handler_enabled(stage_id)

    @classmethod
    def assets_loading(cls) -> bool:
        """Checks if textures are loaded.

        Returns:
            bool: True if textures are loading and not done yet, otherwise False.
        """
        return not SimulationManager._assets_loaded
