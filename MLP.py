import random

class Neuron:
    def __init__(self, n_inputs):
        # Each neuron has 'n_inputs' weights.
        # Every weight is a Value object so it participates in autograd.
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]

        # Bias term, also a Value so it can accumulate gradients.
        self.b = Value(0.0)

    def __call__(self, x):
        # Compute the weighted sum: w1*x1 + w2*x2 + ... + wn*xn + b
        # 'sum' starts from self.b, then adds wi*xi for each input.
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # Apply tanh activation to the result.
        return act.tanh()

    def parameters(self):
        # Return all trainable parameters of this neuron: weights + bias.
        return self.w + [self.b]


class Layer:
    def __init__(self, n_inputs, n_outputs):
        # A layer is simply a list of neurons.
        # Each neuron receives 'n_inputs' inputs.
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        # Forward pass: feed input 'x' to each neuron.
        # Output is a list of neuron outputs.
        return [n(x) for n in self.neurons]

    def parameters(self):
        # Collect parameters from all neurons in the layer.
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, sizes):
        # 'sizes' is a list like [3, 4, 4, 1]
        # meaning:
        #   layer1: 3 inputs → 4 outputs
        #   layer2: 4 inputs → 4 outputs
        #   layer3: 4 inputs → 1 output
        #
        # Build each layer from sizes[i] → sizes[i+1]
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def __call__(self, x):
        # Forward pass through all layers.
        for layer in self.layers:
            x = layer(x)  # output of one layer becomes input of the next

        # If final output is a single Value, return it directly.
        return x[0] if len(x) == 1 else x

    def parameters(self):
        # Collect all parameters from all layers.
        return [p for layer in self.layers for p in layer.parameters()]
