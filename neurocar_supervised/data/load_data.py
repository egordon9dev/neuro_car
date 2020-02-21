import msgpack
import numpy as np


class NeuroCarData:
    def __init__(self, base_path="."):
        self.training_data = []
        self.file_idx = 0
        self.training_idx = 0
        self.file_paths = ["%s/camera_ranges_action/training_data%d.td"%(base_path, i) for i in range(1,81)]
        self.load_new_data_file()
        self.current_frame = None

        if len(self.training_data) == 0:
            sys.exit("file reading failed")

        print("Successfully loaded the training data")
        print("data frames in current file: " + str(len(self.training_data)))
        print("first data frame:\n" + str(len(self.training_data[500][0])))

    def load_new_data_file(self):
        with open(self.file_paths[self.file_idx], "rb") as file:
            self.training_data = msgpack.load(file)
        self.file_idx = (self.file_idx+1) % len(self.file_paths)

    def next_data_frame(self, t):
        image = self.training_data[self.training_idx][0]
        ranges = self.training_data[self.training_idx][1]
        trans_vel = self.training_data[self.training_idx][2]
        rot_vel = self.training_data[self.training_idx][3]
        action = (trans_vel, rot_vel)
        self.current_frame = NeuroCarDataFrame(image, ranges, action)
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

class NeuroCarDataFrame:
    def __init__(self, img, ranges, action):
        self.image = np.asarray(img).reshape((72,128))
        self.ranges = ranges
        self.action = action

    def left_dist(self):
        return ranges[len(ranges)-1]
    def right_dist(self):
        return ranges[0]