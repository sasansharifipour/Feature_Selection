import numpy as np
import pickle

class covariance_operations:

    def __init__(self, file_path, raw_data, feature_cnt):
        self.file_path = file_path
        self.data = raw_data
        self.feature_cnt
    
    def calculate_covariance():

        if (self.data.shape[0] == feature_cnt):
            self.calculated_covariance = np.cov(self.data)

        elif (self.data.shape[1] == feature_cnt):
            self.calculated_covariance = np.cov(self.data.T)

        return self.calculated_covariance

    def save_covariance():
        f = open(self.file_path, 'wb')
        pickle.dump(self.calculated_covariance, f)
        f.close()

    def load_saved_covariance():
        f = open(self.file_path, 'rb')
        self.calculated_covariance = pickle.load(f)
        f.close()
        return data

    def get_calculated_covariance():
        return self.calculated_covariance

    def get_all_id_with_correlation(feature_num, threshold):
        test_list = self.calculated_covariance[feature_num]
        result = [idx for idx, val in enumerate(test_list) if abs(val) > threshold]
        return result
        
        
        
class autoencoder_operations:

    def __init__(self, data, feature_cnt, hidden_layer_size, file_path):
        self.data = data
        self.feature_cnt = feature_cnt
        self.hidden_layer_size
        self.file_path = file_path

    def create_model():
        encoding_dim = self.hidden_layer_size
        input_img = Input(shape=(self.feature_cnt,))
        encoded = Dense(encoding_dim, activation='relu')(input_img)
        decoded = Dense(self.feature_cnt, activation='sigmoid')(encoded)
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        self.model = autoencoder
        return self.model

    def train_model():
        self.model.fit(self.data, self.data,
                epochs=100,
                batch_size=256,
                shuffle=True)

    def save_model():
        self.model.save(self.file_path)

    def load_model():
        self.model = load_model(file_path)

    def get_model():
        return self.model
