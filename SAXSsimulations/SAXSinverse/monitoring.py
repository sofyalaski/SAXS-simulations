import numpy as np
import matplotlib.pyplot as plt



class Visualizer:
    def __init__(self, loss_labels):
            self.n_losses = len(loss_labels)
            self.loss_labels = loss_labels
            self.counter = 0

            self.header = 'Epoch '
            for l in loss_labels:
                self.header += ' %15s' % (l)
                    
    def update_losses(self, losses, logscale=False):
        if self.header:
            print(self.header)
            self.header = None

        print('\r', '    '*20, end='')
        line = '\r%6.3i' % (self.counter)
        for l in losses:
            if logscale: 
                line += '  %14.4f' % (np.log10(l))
            else:
                line += '  %14.4f' % (l)
        print(line)
        self.counter += 1

    def update_images(self, *args):
        pass

    def update_hist(self, *args):
        pass


visualizer = None

def restart():
    global visualizer
    loss_labels = []

    #loss_labels.append('L_ML')
    loss_labels += ['L_fit', 'L_mmd_fwd']
    loss_labels.append('L_mmd_back')
    loss_labels.append('L_reconst')

    loss_labels += [l + '(test)' for l in loss_labels]

    #visualizer = LiveVisualizer(loss_labels)
    visualizer = Visualizer(loss_labels)

def show_loss(losses, logscale=True):
    visualizer.update_losses(losses, logscale)

def show_imgs(*imgs):
    visualizer.update_images(*imgs)

def show_hist(data):
    visualizer.update_hist(data.data.cpu())

def show_cov(data):
    #visualizer.update_cov(data.data.cpu())
    pass

def close():
    visualizer.close()

