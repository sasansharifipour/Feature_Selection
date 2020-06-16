from keras.datasets import mnist
import numpy as np
from CommonCode import autoencoder_operations, covariance_operations, error_calculator

def normalize_dataset(data):
    return data.astype('float32') / 255.

def load_dataset():
    (x_train, _), (x_test, _) = mnist.load_data()
    return x_train, x_test

def create_train_model_obj(data):
    feature_cnt = data.shape[1]
    hidden_layer_size = 10
    model_file_path = "model.h5"
    epochs = 100
    
    auto_encoder = autoencoder_operations( data,
                                           feature_cnt,
                                           hidden_layer_size,
                                           epochs,
                                           model_file_path)
    return auto_encoder

def train_model(auto_encoder):
    auto_encoder.create_model()
    auto_encoder.train_model()
    auto_encoder.save_model()

def load_model(auto_encoder):
    auto_encoder.load_model()
    return auto_encoder.get_model()

def create_covariance_obj(data):
    feature_cnt = data.shape[1]
    covariance_file_path = "covariance.pckl"

    covariance_operator = covariance_operations( data,
                                                 covariance_file_path,
                                                 feature_cnt)
    return covariance_operator
    
def calculate_covariance(covariance_operator):
    covariance_operator.calculate_covariance()
    covariance_operator.save_covariance()

def load_covariance(covariance_operator):
    covariance_operator.load_saved_covariance()
    return covariance_operator.get_calculated_covariance()

def create_error_calculator_obj(train_model_operator, data,
                                covariance_operator):
    feature_cnt = data.shape[1]
    errors_file_path = "errors.pckl"
    threshold = 0.01
    error_calculator_obj = error_calculator( train_model_operator, data, feature_cnt,
                                             errors_file_path, covariance_operator,
                                             threshold)
    
    return error_calculator_obj

def calculate_error(error_calculator_obj):
    error_calculator_obj.calculate_all_errors()
    error_calculator_obj.save_errors()
    
def main():
    x_train, x_test = load_dataset()

    x_train = normalize_dataset(x_train)

    x_test = normalize_dataset(x_test)

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    train_model_operator = create_train_model_obj(x_train)
    #train_model(train_model_operator)
    auto_encoder_model = load_model(train_model_operator)
    
    covariance_operator = create_covariance_obj(x_train)
    #calculate_covariance(covariance_operator)
    calculated_covariance = load_covariance(covariance_operator)

    error_calculator_obj = create_error_calculator_obj(train_model_operator, x_train,
                                                       covariance_operator)
    calculate_error(error_calculator_obj)
    

main()
