# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import List, Optional

import omni.ui as ui
from omni.kit.browser.folder.core import BrowserPropertyDelegate, FileDetailItem


class PropAssetPropertyDelegate(BrowserPropertyDelegate):
    """
    A delegate to show properties of assets of type Prop.
    """

    def accepted(self, items: List[FileDetailItem]) -> bool:
        """BrowserPropertyDelegate method override"""
        return len(items) == 1

    def build_widgets(self, items: List[FileDetailItem]) -> None:
        item = items[0]
        self._container = ui.VStack(height=0, spacing=5)
        with self._container:
            if item.thumbnail:
                self._build_thumbnail(item)
            with ui.HStack():
                ui.Spacer()
                self._name_label = ui.Label(item.name, height=0, style_type_name_override="Asset.Title")
                ui.Spacer()
            item.ui_hook()

    def _build_thumbnail(self, item: FileDetailItem):
        """Builds thumbnail frame and resizes"""
        self._thumbnail_frame = ui.Frame(height=0)
        self._thumbnail_frame.set_computed_content_size_changed_fn(self._on_thumbnail_frame_size_changed)

        with self._thumbnail_frame:
            with ui.HStack():
                ui.Spacer(width=4)
                with ui.ZStack():
                    ui.Rectangle()
                    self._thumbnail_img = ui.Image(
                        item.thumbnail,
                        fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT,
                        alignment=ui.Alignment.CENTER_TOP,
                    )

    def _on_thumbnail_frame_size_changed(self):
        # Dynamic change thumbnail size to be half of frame width
        async def __change_thumbnail_size_async():
            await omni.kit.app.get_app().next_update_async()
            image_size = self._thumbnail_frame.computed_width / 2
            self._thumbnail_img.height = ui.Pixel(image_size)

        asyncio.ensure_future(__change_thumbnail_size_async())


class EmptyPropertyDelegate(BrowserPropertyDelegate):
    """
    A delegate to show when no asset selected.
    """

    def accepted(self, items: Optional[FileDetailItem]) -> bool:
        """BrowserPropertyDelegate method override"""
        return len(items) == 0

    def build_widgets(self, items: Optional[FileDetailItem]) -> None:
        """BrowserPropertyDelegate method override"""
        ui.Label("Please Select a Isaac Example!", alignment=ui.Alignment.CENTER)


class MultiPropertyDelegate(BrowserPropertyDelegate):
    """
    A delegate to show when multiple items are selected.
    """

    def accepted(self, items: List[FileDetailItem]) -> bool:
        """BrowserPropertyDelegate method override"""
        return len(items) > 1

    def build_widgets(self, items: List[FileDetailItem]) -> None:
        """BrowserPropertyDelegate method override"""
        label_text = f"Multiple Isaac Assets Selected [{len(items)}]"
        ui.Label(label_text, alignment=ui.Alignment.CENTER)
