import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import keras
from data import load_data

# This script will display a video of the training data images
# requires ffmpeg `sudo apt install ffmpeg`
img_width, img_height = (128, 72)
min_range, max_range = (0.08, 100.0)
def preprocess_label(l):
    # clamp label to within bounds
    l = min(max(l, min_range), max_range)
    # normalize
    l = (l-min_range)/(max_range-min_range)
    # de-nan/inf
    if np.isnan(l) or np.isinf(l):
        l = 1
    return l

range_idxs = [0,1,35,36]
models = [keras.models.load_model("./ncmodel%d.h5"%(i)) for i in range_idxs]

ncdata = load_data.NeuroCarData(paths=["./data/grayscale_data.td"])
fig = plt.figure(figsize=(8,8))

fps = 3
duration = 30
image = plt.imshow(ncdata.next_data_frame(0)[0].reshape(img_height,img_width).tolist(), interpolation='none', aspect='auto', vmin=0, vmax=255)

def animate(i):
    frame = ncdata.next_data_frame_unprocessed(0)
    img = np.asarray(frame[0])
    image.set_array(img.reshape((img_height,img_width)).tolist())
    labels = [preprocess_label(frame[1][i]) for i in range_idxs]
    img = load_data.process_image(img).reshape((img.shape[0], img.shape[1], 1))
    model_input = np.array([img])
    model_outputs = [m.predict(model_input)[0] for m in models]
    print("Networks Output (prediction / true-value):")
    for i in range(len(range_idxs)):
        print("\trange %d:  %.3f / %.3f "%(range_idxs[i], model_outputs[i], labels[i]))
    print("\n")
    return [image]

anim = animation.FuncAnimation(fig, animate, frames=400, interval = 1000/fps)

# anim.save('test_anim.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
plt.show()