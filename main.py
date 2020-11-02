import numpy as np
from utils import load_dataset
from model import Layer, ReLU, Network
import matplotlib.pyplot as plt
from tqdm import trange

_input, _output, X_val, y_val = load_dataset()

network = Network()
network.add_layer(Layer(_input.shape[1], 600))
network.add_layer(ReLU())
network.add_layer(Layer(600,400))
network.add_layer(ReLU())
network.add_layer(Layer(400,300))
network.add_layer(ReLU())
network.add_layer(Layer(300,200))
network.add_layer(ReLU())
network.add_layer(Layer(200,10))

# Funcion para sacar extractos de imagenes de forma aleatoria para entrenar la red
def iterate_minibatches(inputs, targets, batchsize):
    indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]


epochs = 10
batch_size = 120
train_log = []
val_log = []

for epoch in range(epochs):
    for x_batch, y_batch in iterate_minibatches(_input, _output, batch_size):
        network.train(x_batch,y_batch)

    train_log.append(np.mean(network.predict(_input)==_output))
    val_log.append(np.mean(network.predict(X_val)==y_val))

plt.plot(train_log, label='train accuracy')
plt.plot(val_log, label='val accuracy')
plt.legend(loc='best')
plt.grid()
plt.show()
plt.savefig('train.png')