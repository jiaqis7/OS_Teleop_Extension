# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import List, Optional

import omni.ui as ui
from omni.kit.browser.folder.core import BrowserPropertyDelegate, FileDetailItem


class EmptyPropertyDelegate(BrowserPropertyDelegate):
    """
    A delegate to show when no asset selected.
    """

    def accepted(self, items: Optional[FileDetailItem]) -> bool:
        """BrowserPropertyDelegate method override"""
        return len(items) == 0

    def build_widgets(self, items: Optional[FileDetailItem]) -> None:
        """BrowserPropertyDelegate method override"""
        ui.Label("Please Select a Isaac Asset!", alignment=ui.Alignment.CENTER)
