import importlib
import os
import time

import modules
import filewatcher

def main():
    current_dir = os.path.dirname(__file__)
    p = os.path.join(current_dir, "modules.py")
    w = filewatcher.FileModifiedTimeTracker(p)
    w.did_read_file_just_now()
    while True:
        if w.has_changes:
            s = time.time()
            print("HAS CHANGES")
            importlib.reload(modules)
            modules.BabiesFirstSynthie()
            print(time.time()-s)
            print(vars(modules.StepSequencing)["VAR"])
            w.did_read_file_just_now()
        time.sleep(1)


if __name__ == '__main__':
    main()