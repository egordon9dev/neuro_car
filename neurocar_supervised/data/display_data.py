import load_data
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation

# This script will display a video of the training data images
# requires ffmpeg `sudo apt install ffmpeg`

ncdata = load_data.NeuroCarData()
fig = plt.figure(figsize=(8,8))

fps = 30
duration = 30
image = plt.imshow(ncdata.next_data_frame(0).image, interpolation='none', aspect='auto', vmin=0, vmax=1)

def animate(i):
    image.set_array(ncdata.next_data_frame(0).image.tolist())
    return [image]

anim = animation.FuncAnimation(fig, animate, frames=fps * duration, interval = 1000/fps)

# anim.save('test_anim.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
plt.show()