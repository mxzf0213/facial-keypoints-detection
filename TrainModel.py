from Model import buildModel
from keras.models import load_model
import os
from DataProcess import load2d
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import LearningRateScheduler

if os.path.exists('CnnModel.h5'):
    model = load_model('CnnModel.h5')
else:
    model = buildModel()

batch_size = 128
epochs = 1000
verbose = 1

x_train, y_train = load2d(test=False, cols=True)


def mycallback(epoch):
    lr = K.get_value(model.optimizer.lr)
    if epoch % 100 == 0 and epoch != 0 and lr > 0.0001:
        base = 0.5
        K.set_value(model.optimizer.lr, lr * base)
    return K.get_value(model.optimizer.lr)


reduce_lr = LearningRateScheduler(mycallback)

# K.set_value(model.optimizer.lr, 0.01)
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[reduce_lr])

history_dict = history.history
loss_values = history_dict['loss']

fig = plt.figure()

ax1 = fig.add_subplot(111)  # 一行一列一块
epochs = range(1, len(loss_values) + 1)
ax1.plot(epochs, loss_values, 'b', label='Training Loss')
plt.show()

loss_values = loss_values[100:]
epochs = range(1, len(loss_values) + 1)
plt.clf()
plt.plot(epochs, loss_values, 'b', label='Training Loss')
plt.show()

model.save('CnnModel.h5')
