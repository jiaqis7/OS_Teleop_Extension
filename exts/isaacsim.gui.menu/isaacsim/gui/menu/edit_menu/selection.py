"""This module provides the Selection class to encapsulate selection details, including a description and a timestamp."""

__all__ = ["Selection"]

import time


class Selection:
    """A class that encapsulates a selection with a description.

    This class holds the details of a selection, including a timestamp of when the selection was created or modified. It provides the ability to update the timestamp to the current time.

    Args:
        description (str): A brief description of the selection.
        selection: The actual content of the selection."""

    def __init__(self, description, selection):
        """Initialize a new Selection instance with the given description and selection data."""
        self.time = time.monotonic()
        self.description = description
        self.selection = selection

    def touch(self):
        """Updates the timestamp of the selection to the current time."""
        self.time = time.monotonic()
