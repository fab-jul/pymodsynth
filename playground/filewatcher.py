"""Watch last modified date of files."""

import os
import time


class FileModifiedTimeTracker:
    """Simple tracker.

    Note that this tracker does not implement any polling, the caller has to poll.

    Usage:

      t = FileModifiedTimeTracker(path)
      # Some sort of event loop.
      while True:
        if t.has_changes:
          # Do something with the file
          ...
          # Mark as used
          t.did_read_file_just_now()
    """

    def __init__(self, path_to_watch: str):
        if not os.path.isfile(path_to_watch):
            raise FileNotFoundError(path_to_watch)
        self.path_to_watch = path_to_watch
        self.known_last_change = None

    def did_read_file_just_now(self):
        """Call if you just read the file to update the last change date."""
        self.known_last_change = os.stat(self.path_to_watch).st_mtime

    @property
    def has_changes(self) -> bool:
        """Call to check if anything changed since you last read."""
        last_change = os.stat(self.path_to_watch).st_mtime
        return last_change != self.known_last_change


def _test():
    with open("test.txt", "w") as f:
        f.write("TEST")
    f = FileModifiedTimeTracker("test.txt")
    while True:
        print("Has changes?", f.has_changes)
        time.sleep(1)


if __name__ == '__main__':
    _test()
