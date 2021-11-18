import importlib
import os
import pprint
import time

import modules
import filewatcher

# LIMITATION:
# - unnamed stuff not caught

def main():
    current_dir = os.path.dirname(__file__)
    p = os.path.join(current_dir, "modules.py")
    w = filewatcher.FileModifiedTimeTracker(p)
    prev_m = None
    while True:
        if w.has_changes:
            s = time.time()
            print("HAS CHANGES")
            if prev_m is not None:
                src_params = prev_m.find_params()
                src_state = prev_m.find_state()
            importlib.reload(modules)  # RELOAD CODE!
            m = modules.TestModule()
            if prev_m is not None:
                m.copy_params_and_state_from(
                    src_params=src_params,
                    src_state=src_state)
            prev_m = m
            m.freq0.value += 1
            print(time.time()-s)
            w.did_read_file_just_now()
        time.sleep(1)


if __name__ == '__main__':
    main()