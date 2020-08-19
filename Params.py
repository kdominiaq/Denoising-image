class Params:
    '''
    Parent class for Test class
    also stores variables for train.py
    '''
    def __init__(self):
        self.learning_rate = 0.01
        self.num_epochs = 100
        self.num_train_images_in_epoch = 5000
        self.num_test_images = 10
        self.model_save_PATH = "./denoised_model.pth"

