import numpy as np

class ABCLayer:
    def forward(self, input):
        pass

    def backward(self, input, grad_output):
        pass


class ReLU(ABCLayer):
    def forward(self, input):
        relu = np.maximum(0, input)
        return relu

    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output*relu_grad


class Layer:
    def __init__(self, input_units, output_units, learning_rate=0.01):
        self.learning_rate = learning_rate

        # https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
        self.weights = np.random.normal(loc=0.0,
                                        scale = np.sqrt(2/(input_units+output_units)),
                                        size = (input_units,output_units)) # Se generan pesos randoms
        self.biases = np.zeros(output_units) # Se dejan los bias en 0 del mismo tamaño del output

    def forward(self, input):
        """
        f(x) = <W*x> + b
        """
        return np.dot(input, self.weights) + self.biases

    def backward(self,input,grad_output):
        # https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/
        grad_input = np.dot(grad_output, self.weights.T)

        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0)*input.shape[0]

        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input


class Network:

    def __init__(self):
        self.network = []


    def add_layer(self, layer):
        self.network.append(layer)

    def forward(self, X):
        """
        Recoremos todas las capas activando con la funcion forward de cada una
        """
        activations = []
        input = X

        for l in self.network:
            activations.append(l.forward(input))
            # El output de la ultima capa es el input de la siguiente
            input = activations[-1]

        return activations

    def predict(self, X):
        logits = self.forward(X)[-1] # Ultima capa
        return logits.argmax(axis=-1) # El indice del mayor valor de la matriz

    def train(self, X, y):
        """
        Entrena la red
        """

        # Activa las capas de la red
        layer_activations = self.forward(X)
        layer_inputs = [X] + layer_activations  # layer_inputs[i] es un input para la red [i]
        logits = layer_activations[-1]

        # se calcula el softmax
        loss = self.softmax_crossentropy_with_logits(logits, y)
        loss_grad = self.grad_softmax_crossentropy_with_logits(logits, y)

        # Se hace el backpropagation
        for layer_index in range(len(self.network))[::-1]:
            layer = self.network[layer_index]
            loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)

        return np.mean(loss)

    def softmax_crossentropy_with_logits(self, logits, reference_answers):
        logits_for_answers = logits[np.arange(len(logits)), reference_answers]

        xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))

        return xentropy

    def grad_softmax_crossentropy_with_logits(self, logits, reference_answers):
        ones_for_answers = np.zeros_like(logits)
        ones_for_answers[np.arange(len(logits)), reference_answers] = 1

        softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)

        return (- ones_for_answers + softmax) / logits.shape[0]
