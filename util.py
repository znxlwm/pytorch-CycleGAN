import itertools, imageio, torch, random
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from scipy.misc import imresize
from torch.autograd import Variable

def show_result(G, x_, y_, num_epoch, show = False, save = False, path = 'result.png'):
    test_images = G(x_)

    size_figure_grid = 3
    fig, ax = plt.subplots(x_.size()[0], size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(x_.size()[0]), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for i in range(x_.size()[0]):
        ax[i, 0].cla()
        ax[i, 0].imshow((x_[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[i, 1].cla()
        ax[i, 1].imshow((test_images[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[i, 2].cla()
        ax[i, 2].imshow((y_[i].numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_A_losses']))

    y1 = hist['D_A_losses']
    y2 = hist['D_B_losses']
    y3 = hist['G_A_losses']
    y4 = hist['G_B_losses']
    y5 = hist['A_cycle_losses']
    y6 = hist['B_cycle_losses']


    plt.plot(x, y1, label='D_A_loss')
    plt.plot(x, y2, label='D_B_loss')
    plt.plot(x, y3, label='G_A_loss')
    plt.plot(x, y4, label='G_B_loss')
    plt.plot(x, y5, label='A_cycle_loss')
    plt.plot(x, y6, label='B_cycle_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def generate_animation(root, model, opt):
    images = []
    for e in range(opt.train_epoch):
        img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(root + model + 'generate_animation.gif', images, fps=5)

def data_load(path, subfolder, transform, batch_size, shuffle=False):
    dset = datasets.ImageFolder(path, transform)
    ind = dset.class_to_idx[subfolder]

    n = 0
    for i in range(dset.__len__()):
        if ind != dset.imgs[n][1]:
            del dset.imgs[n]
            n -= 1

        n += 1

    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)

def imgs_resize(imgs, resize_scale = 286):
    outputs = torch.FloatTensor(imgs.size()[0], imgs.size()[1], resize_scale, resize_scale)
    for i in range(imgs.size()[0]):
        img = imresize(imgs[i].numpy(), [resize_scale, resize_scale])
        outputs[i] = torch.FloatTensor((img.transpose(2, 0, 1).astype(np.float32).reshape(-1, imgs.size()[1], resize_scale, resize_scale) - 127.5) / 127.5)

    return outputs

def random_crop(imgs, crop_size = 256):
    outputs = torch.FloatTensor(imgs.size()[0], imgs.size()[1], crop_size, crop_size)
    for i in range(imgs.size()[0]):
        img = imgs[i]
        rand1 = np.random.randint(0, imgs.size()[2] - crop_size)
        rand2 = np.random.randint(0, imgs.size()[2] - crop_size)
        outputs[i] = img[:, rand1: crop_size + rand1, rand2: crop_size + rand2]

    return outputs

def random_fliplr(imgs):
    outputs = torch.FloatTensor(imgs.size())
    for i in range(imgs.size()[0]):
        if torch.rand(1)[0] < 0.5:
            img = torch.FloatTensor(
                (np.fliplr(imgs[i].numpy().transpose(1, 2, 0)).transpose(2, 0, 1).reshape(-1, imgs.size()[1], imgs.size()[2], imgs.size()[3]) + 1) / 2)
            outputs[i] = (img - 0.5) / 0.5
        else:
            outputs[i] = imgs[i]

    return outputs

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

class image_store():
    def __init__(self, store_size=50):
        self.store_size = store_size
        self.num_img = 0
        self.images = []

    def query(self, image):
        select_imgs = []
        for i in range(image.size()[0]):
            if self.num_img < self.store_size:
                self.images.append(image)
                select_imgs.append(image)
                self.num_img += 1
            else:
                prob = np.random.uniform(0, 1)
                if prob > 0.5:
                    ind = np.random.randint(0, self.store_size - 1)
                    select_imgs.append(self.images[ind])
                    self.images[ind] = image
                else:
                    select_imgs.append(image)

        return Variable(torch.cat(select_imgs, 0))

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images