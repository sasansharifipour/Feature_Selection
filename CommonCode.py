import numpy as np

class covariance_operations:

    def __init__(self, file_path, raw_data, feature_cnt):
        self.file_path = file_path
        self.data = raw_data
        self.feature_cnt
    
    def calculate_covariance():

        if (self.data.shape[0] == feature_cnt)
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
        
        
        
