# Denoising an image by deep learning - comparison of neural network architectures
## Results
You can check the results of the neural network [here](https://drive.google.com/file/d/1KY74Q58q1Ef6QwV9gf4DcoKUkSit9oo7/view?usp=sharing).

##Introduction
The environment in which the project will be run must support  **cuda**, otherwise the program will throw an error.
Project contains three types of structures which are used in machine learning for denoising images, which are:
- Autoencoder,
- UNet,
- Resnet

Conda environment for this project can be downloaded from [here](https://drive.google.com/drive/folders/152XI1wCo6CfD2vwqND4denkJqtDSfhFG?usp=sharing "Conda environment").

## Setup
#### Parameters
You can change all parameters of learning in **Params.py** file, like:
- learning rate,
- numbers of epochs,
- number of test and train images.

You can also decide to save an image during training or after the test.
#### Dataset
First, you must decide which dataset you will use in your session. It is posible by changing:
- Line 53 in **Train.py**:
```python
        train_load = Dataset.CIFAR_train_loader()
```
- Line  44 in **Test.py**
```python
         for i, data in enumerate(Dataset.CIFAR_test_loader()):
```

#### Train 
Second, you must decide which module you will use to train:
- Line 49 in **Train.py**
```python
        model = Resnet().cuda()
```

also you have to remember to change the path to save the model:
- Line 81 in **Train.py**
```python
        torch.save(model, self.Resnet_model_save_PATH)
```

#### Test
Finally you can observe the results from net.  Remember to configure path of saving images if you decided to save it:
- Line 63 in **Test.py**:
```python
        image_save(combined_img, f"./test_image/Resnet/test_img_{i + 1}.png")
```



