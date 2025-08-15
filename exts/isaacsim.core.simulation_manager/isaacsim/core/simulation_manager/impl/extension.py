import carb
import omni.ext
import omni.kit.app

from .. import _simulation_manager
from .isaac_events import IsaacEvents
from .simulation_manager import SimulationManager

# expose pybind interface/API
_simulation_manager_interface = None


def acquire_simulation_manager_interface():
    return _simulation_manager_interface


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id):
        ext_path = omni.kit.app.get_app().get_extension_manager().get_extension_path(ext_id)
        # acquire the pybind interface
        global _simulation_manager_interface
        _simulation_manager_interface = _simulation_manager.acquire_simulation_manager_interface()
        SimulationManager._simulation_manager_interface = _simulation_manager_interface
        SimulationManager._initialize()

    def on_shutdown(self):
        # release the pybind interface
        global _simulation_manager_interface
        SimulationManager._clear()
        _simulation_manager.release_simulation_manager_interface(_simulation_manager_interface)
        _simulation_manager_interface = None
