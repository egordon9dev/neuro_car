import msgpack
import numpy as np

def process_image(image):
    # normalize image to values in [0,1]
    image = np.asarray(image) / 255
    # subtract mean pixel value
    image = image - np.mean(image)
    return image

class NeuroCarData:
    def __init__(self, paths, file_idx=0, training_idx=0):
        self.training_data = []
        self.file_idx = file_idx
        self.training_idx = training_idx
        self.file_paths = paths
        self.load_new_data_file()
        self.current_frame = None

        # print("Successfully loaded the training data")
        # print("data frames in current file: " + str(len(self.training_data)))
        # print("first data frame:\n" + str(len(self.training_data[500][0])))

    def load_new_data_file(self):
        with open(self.file_paths[self.file_idx], "rb") as file:
            self.training_data = msgpack.load(file)
        self.file_idx = (self.file_idx+1) % len(self.file_paths)

    def next_data_frame_unprocessed(self, t):
        image = self.training_data[self.training_idx][0]
        ranges = self.training_data[self.training_idx][1]
        self.current_frame = (image, ranges)
        self.training_idx += 1
        if self.training_idx >= len(self.training_data):
            self.load_new_data_file()
            self.training_idx = 0
        return self.current_frame

    def next_data_frame(self, t):
        image = process_image(self.training_data[self.training_idx][0])
        ranges = self.training_data[self.training_idx][1]
        self.current_frame = (image, ranges)
        self.training_idx += 1
        if self.training_idx >= len(self.training_data):
            self.load_new_data_file()
            self.training_idx = 0
        return self.current_frame


    # TODO if we want to use these actions, normalize them here first
    # def downscale_action(self, x):
    #     return np.array([x[0]/4, x[1]/1.5])
    # def upscale_action(self, x):
    #     return np.array([x[0]*4, x[1]*1.5])
    
    # def get_optimal_action(self, x):
    #     optimal_action = self.downscale_action(np.array([self.training_data[training_idx][2], self.training_data[training_idx][3]]))
    #     return optimal_action