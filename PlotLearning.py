import matplotlib.pyplot as plt
import keras

class PlotLearning(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.logs = []
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 5))
        plt.show()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.val_accuracy.append(logs.get('val_accuracy'))
        self.i += 1

        # clear the old figure
        for ax in self.axes:
            ax.clear()

        # plot lossself.axes[0].plot(self.x, self.losses, label='loss')
        self.axes[0].plot(self.x, self.val_losses, label='val_loss')
        self.axes[0].legend()
        
        # plot accuracy
        self.axes[1].plot(self.x, self.accuracy, label='accuracy')
        self.axes[1].plot(self.x, self.val_accuracy, label='val_accuracy')
        self.axes[1].legend()

        plt.draw()
        plt.pause(0.001)
