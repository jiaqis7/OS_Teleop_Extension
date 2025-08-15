# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
import sys
from typing import List

PACKAGES_PATH = []
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# add packages to sys.path
with open(os.path.join(SCRIPT_DIR, "packages.txt"), "r") as f:
    for p in f.readlines():
        p = p.strip()
        if p:
            PACKAGES_PATH.append(p)
            if p not in sys.path:
                print("Adding package to sys.path: {}".format(p))
                sys.path.append(p)

# add provisioners to sys.path
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..", "provisioners")))


from jupyter_client.kernelspec import KernelSpecManager as _KernelSpecManager


class KernelSpecManager(_KernelSpecManager):
    def __init__(self, *args, **kwargs):
        """Custom kernel spec manager to allow for loading of custom kernels"""
        super().__init__(*args, **kwargs)
        kernel_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "kernels"))
        if kernel_dir not in self.kernel_dirs:
            self.kernel_dirs.append(kernel_dir)


def main(ip: str = "0.0.0.0", port: int = 8228, argv: List[str] = [], classic_notebook_interface: bool = False) -> None:
    """Entry point for launching Jupyter Notebook/Lab

    :param ip: Notebook server IP address (default: "0.0.0.0")
    :type code: str, optional
    :param port: Notebook server port number (default: 8228)
    :type code: int, optional
    :param argv: Command line arguments to pass to Jupyter Notebook/Lab (default: [])
    :type code: List of strings, optional
    :param classic_notebook_interface: Whether to use the classic notebook interface (default: False)
                                       If false, the Jupyter Lab interface will be used
    :type code: bool, optional
    """

    def load_jupyter_notebook():
        try:
            from notebook.notebookapp import NotebookApp

            app = NotebookApp(ip=ip, port=port, kernel_spec_manager_class=KernelSpecManager)
            app.initialize(argv=argv)
        except ModuleNotFoundError:
            app = load_jupyter_lab(classic=True)
        return app

    def load_jupyter_lab(classic=False):
        from jupyter_server.serverapp import ServerApp

        if classic:
            from nbclassic.notebookapp import NotebookApp as LabApp
        else:
            from jupyterlab.labapp import LabApp

        jpserver_extensions = {LabApp.get_extension_package(): True}
        find_extensions = LabApp.load_other_extensions
        if "jpserver_extensions" in LabApp.serverapp_config:
            jpserver_extensions.update(LabApp.serverapp_config["jpserver_extensions"])
            LabApp.serverapp_config["jpserver_extensions"] = jpserver_extensions
            find_extensions = False

        app = ServerApp.instance(
            ip=ip, port=port, kernel_spec_manager_class=KernelSpecManager, jpserver_extensions=jpserver_extensions
        )

        app.aliases.update(LabApp.aliases)
        app.initialize(argv=argv, starter_extension=LabApp.name, find_extensions=find_extensions)
        return app

    # load app
    app = load_jupyter_notebook() if classic_notebook_interface else load_jupyter_lab()

    # write url to file in script directory
    with open(os.path.join(SCRIPT_DIR, "notebook.txt"), "w") as f:
        f.write(app.display_url)

    app.start()


if __name__ == "__main__":

    argv = sys.argv[1:]

    # testing the launcher
    if not len(argv):
        print("Testing the launcher")
        argv = [
            "0.0.0.0",  # ip
            "8888",  # port
            "",  # token
            "False",  # classic_notebook_interface
            "",  # notebook_dir
            "--allow-root --no-browser",
        ]  # extra arguments

    # get function arguments
    ip = argv[0]
    port = int(argv[1])
    token = argv[2]
    classic_notebook_interface = argv[3] == "True"

    # get notebook_dir
    if not argv[4]:
        notebook_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "notebooks"))
        notebook_dir = "--notebook-dir={}".format(notebook_dir)

    # get token
    if classic_notebook_interface:
        token = "--NotebookApp.token={}".format(token)
    else:
        token = "--ServerApp.token={}".format(token)

    # assets path
    app_dir = []
    if not classic_notebook_interface:
        for p in PACKAGES_PATH:
            print("Checking package to app_dir: {}".format(p))
            if os.path.exists(os.path.join(p, "share", "jupyter", "lab")):
                app_dir = ["--app-dir={}".format(os.path.join(p, "share", "jupyter", "lab"))]
            if os.path.exists(os.path.join(p, "jupyterlab", "static")):
                app_dir = ["--app-dir={}".format(os.path.join(p, "jupyterlab"))]
            if app_dir:
                break
        print("app_dir: {}".format(app_dir))

    # clean up the argv
    argv = app_dir + [token] + [notebook_dir] + argv[5].split(" ")

    # run the launcher
    print("Starting Jupyter {} at {}:{}".format("Notebook" if classic_notebook_interface else "Lab", ip, port))
    print(" with argv: {}".format(" ".join(argv)))

    main(ip=ip, port=port, argv=argv, classic_notebook_interface=classic_notebook_interface)

    # delete notebook.txt
    try:
        os.remove(os.path.join(SCRIPT_DIR, "notebook.txt"))
    except:
        pass
