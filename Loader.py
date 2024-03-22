import os
import numpy as np
import cv2
import random

# main_dir is a folder that contains videos and labels folder
# dim is used to reduce the spatial size of frames
# Inside the main_dir two folders will be existing with the names of "thermal", and "resp". 
# The thermal folder contains .npy files of videos
# The resp folder contains the .npy files for ground truth signal
# The name of .npy files for the same sample of thermal and respiratory data will have the same name.


class ObjTrainLoader():

    def __init__(self, main_dir, dim=(64, 64), transform=None):  #dim=(64, 64)
        data_dir = os.path.join(main_dir, "thermal")
        labels_dir = os.path.join(main_dir, "resp")
        data_name = os.listdir(data_dir)
        random.shuffle(data_name)
        #labels_name = os.listdir(labels_dir)
        self.data_name = data_name
        #self.labels_name = labels_name
        self.n_samples = len(data_name)
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.dim = dim

    def __getitem__(self, index):
        loading_data_dir = os.path.join(self.data_dir, self.data_name[index])
        loading_label_dir = os.path.join(
            self.labels_dir, self.data_name[index])
        loaded_video = np.load(loading_data_dir)
        transformed_video = np.zeros((loaded_video.shape[0], self.dim[0], self.dim[1]))

        for i, frame in enumerate(loaded_video):
            resized_frame = cv2.resize(frame, self.dim, interpolation = cv2.INTER_AREA)
            transformed_video[i] = resized_frame

        transformed_video = transformed_video/255.0
        transformed_video = (transformed_video - 0.1034)/(0.0190)
        transformed_video = np.repeat(transformed_video[np.newaxis, ...], 3, axis=0)
        


        label = np.load(loading_label_dir)
        label = (label - np.mean(label))/np.std(label)

        loaded_label = label
        sample = transformed_video, loaded_label
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples