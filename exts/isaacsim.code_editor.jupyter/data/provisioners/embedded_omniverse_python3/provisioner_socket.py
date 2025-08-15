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
from typing import Any, List

from jupyter_client.connect import KernelConnectionInfo
from jupyter_client.launcher import launch_kernel
from jupyter_client.provisioning import LocalProvisioner


class Provisioner(LocalProvisioner):
    async def launch_kernel(self, cmd: List[str], **kwargs: Any) -> KernelConnectionInfo:
        # set paths
        if sys.platform == "win32":
            cmd[0] = os.path.abspath(os.path.join(os.path.dirname(os.__file__), "..", "python.exe"))
        else:
            cmd[0] = os.path.abspath(os.path.join(os.path.dirname(os.__file__), "..", "..", "bin", "python3"))
        cmd[1] = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "launchers", "ipykernel_launcher.py")
        )

        self.log.info("Launching kernel: %s", " ".join(cmd))

        scrubbed_kwargs = LocalProvisioner._scrub_kwargs(kwargs)
        self.process = launch_kernel(cmd, **scrubbed_kwargs)
        pgid = None
        if hasattr(os, "getpgid"):
            try:
                pgid = os.getpgid(self.process.pid)
            except OSError:
                pass

        self.pid = self.process.pid
        self.pgid = pgid
        return self.connection_info
