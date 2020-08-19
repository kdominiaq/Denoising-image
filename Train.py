import torch
from Dataset import Dataset
from torch import nn
from skimage.util import random_noise
from Autoencoder import Autoencoder
from Params import Params


class Train(Params):
    def __init__(self, device):
        super().__init__()
        model = Autoencoder().cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        train_load = Dataset.train_loader()
        for epoch in range(self.num_epochs):
            for i, data in enumerate(train_load):
                clean_img_train, _ = data[0], data[1]
                noised_img_train = torch.tensor(
                    random_noise(clean_img_train, mode='s&p', salt_vs_pepper=0.5, clip=True))
                # fixing "RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same"
                clean_img_train, noised_img_train = clean_img_train.to(device), noised_img_train.to(device)
                output = model(noised_img_train)
                loss = criterion(output, clean_img_train)
                # backward
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # period of the number of photos in dateset
                if i == self.num_train_images_in_epoch:
                    break
                # log
            print(f'epoch [{epoch + 1}/{self.num_epochs}], loss:{loss.item():.4f}')

            # saving image during learning
            # if epoch % 10 == 0:
            #    combined_img = torch.cat((clean_img_train, noised_img_train, output), 3) #combining images
            #    image_save(combined_img, f"./{epoch}_all_vs1.png")

        torch.save(model, self.model_save_PATH)