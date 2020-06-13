import os
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataManager:
    @staticmethod
    def load_data(img_file_name, labels_file_name):
        """
        Récupère des données (format .npy)
        retourne deux arrays X et Y ainsi que leurs tailles respectives
        """
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
        x = np.load(os.path.join(data_dir, img_file_name))
        y = np.load(os.path.join(data_dir, labels_file_name))

        size_x = x.shape[0]
        size_y = y.shape[0]

        if size_x != size_y:
            print(img_file_name, size_x)
            print(labels_file_name, size_y)
            raise ValueError('Set error : image and labels numbers do not match')
        return x, y, size_x, size_y

    @staticmethod
    def feature_scaling(x_trn, x_dev):
        """
        Initialise un StandardScaler par rapport au training set
        Puis applique la transformation sur le training set et au test set
        """
        scaler = StandardScaler().fit(x_trn)

        x_trn = scaler.transform(x_trn)
        x_dev = scaler.transform(x_dev)
        return x_trn, x_dev
