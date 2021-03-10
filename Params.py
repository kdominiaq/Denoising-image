class Params:
    '''
    Parent class for Test class
    also stores variables for train.py
    '''
    def __init__(self):
        self.learning_rate = 0.01
        self.num_epochs = 10
        self.num_train_images_in_epoch = 5000
        self.num_test_images = 530
        self.AU_model_save_PATH = "./AU_denoised_model.pth"
        self.UNet_model_save_PATH = "./UNet_denoised_model.pth"
        self.Resnet_model_save_PATH = "./Resnet_denoised_model.pth"



