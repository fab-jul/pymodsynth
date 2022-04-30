import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
from random import random

print(matplotlib.__version__)

# set up the figure
plt.ion()
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.set_xlabel('Time')
ax.set_ylabel('Value')
t = []
y = []
ax.plot(t, y, 'ko-', markersize=10)  # add an empty line to the plot
fig.show()  # show the window (figure will be in foreground, but the user may move it to background)

# plot things while new data is generated:
# (avoid calling plt.show() and plt.pause() to prevent window popping to foreground)
t0 = time.time()
while True:
    t.append(time.time() - t0)  # add new x data value
    y.append(random())  # add new y data value
    ax.lines[0].set_data(t, y)  # set plot data
    ax.relim()  # recompute the data limits
    ax.autoscale_view()  # automatic axis scaling
    fig.canvas.flush_events()  # update the plot and take care of window events (like resizing etc.)
    time.sleep(1)  # wait for next loop iteration
