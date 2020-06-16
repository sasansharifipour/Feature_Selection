import numpy as np
import pickle

from keras.layers import Input, Dense
from keras.models import Model, load_model

class covariance_operations:

    def __init__(self, raw_data, file_path, feature_cnt):
        self.file_path = file_path
        self.data = raw_data
        self.feature_cnt = feature_cnt
    
    def calculate_covariance(self):

        if (self.data.shape[0] == self.feature_cnt):
            self.calculated_covariance = np.cov(self.data)

        elif (self.data.shape[1] == self.feature_cnt):
            self.calculated_covariance = np.cov(self.data.T)

        return self.calculated_covariance

    def save_covariance(self):
        f = open(self.file_path, 'wb')
        pickle.dump(self.calculated_covariance, f)
        f.close()

    def load_saved_covariance(self):
        f = open(self.file_path, 'rb')
        self.calculated_covariance = pickle.load(f)
        f.close()

    def get_calculated_covariance(self):
        return self.calculated_covariance

    def get_all_id_with_correlation(self, feature_num, threshold):
        test_list = self.calculated_covariance[feature_num]
        result = [idx for idx, val in enumerate(test_list) if abs(val) > threshold]
        return result
        
class autoencoder_operations:

    def __init__(self, data, feature_cnt, hidden_layer_size, epochs, file_path):
        self.data = data
        self.feature_cnt = feature_cnt
        self.hidden_layer_size = hidden_layer_size
        self.epochs = epochs
        self.file_path = file_path

    def create_model(self):
        encoding_dim = self.hidden_layer_size
        input_img = Input(shape=(self.feature_cnt,))
        encoded = Dense(encoding_dim, activation='relu')(input_img)
        decoded = Dense(self.feature_cnt, activation='sigmoid')(encoded)
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        self.model = autoencoder
        return self.model

    def train_model(self):
        self.model.fit(self.data, self.data,
                epochs=self.epochs,
                batch_size=256,
                shuffle=True)

    def save_model(self):
        self.model.save(self.file_path)

    def load_model(self):
        self.model = load_model(self.file_path)

    def get_model(self):
        return self.model

    def calculate_predict(self, data):
        return self.model.predict(data)

class error_calculator:

    def __init__(self, model, data, feature_cnt, file_path, covariance_operator, threshold):
        self.model = model
        self.data = data
        self.file_path = file_path
        self.feature_cnt = feature_cnt
        self.covariance_operator = covariance_operator
        self.threshold = threshold
        self.error = np.zeros(feature_cnt)
        self.actual_error = self.calculate_error(self.calculate_result(self.data))
                                      
    def calculate_result(self, data):
        return self.model.calculate_predict(data)

    def calculate_error(self, predicted_data):
        return np.sum(abs(self.data - predicted_data))

    def calculate_result_by_removing_features( self, data,
                                               feature_index,
                                               removing_features):
        data[:, feature_index] = 0
        data[:, removing_features] = 0

        return self.model.calculate_predict(data)
    
    def calculate_error_by_removing_features( self, feature_index, removing_features):

        result = self.calculate_result_by_removing_features( self.data,
                                          feature_index, removing_features)
    
        return (self.calculate_error(result) - self.actual_error) / (len(removing_features) + 1)

    def calculate_all_errors(self):

        #actual_result = calculate_result(auto_encoder, data)
        #actual_error = calculate_error(data, actual_result)
        
        for i in range(0, self.feature_cnt):
            res = self.covariance_operator.get_all_id_with_correlation(i, self.threshold)
            print(i)
            self.error[i] = self.calculate_error_by_removing_features(i, res)

    def save_errors(self):
        f = open(self.file_path, 'wb')
        pickle.dump(self.error, f)
        f.close()

    def load_saved_errors(self):
        f = open(self.file_path, 'rb')
        self.error = pickle.load(f)
        f.close()
