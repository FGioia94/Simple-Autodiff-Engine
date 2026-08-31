class Value:
    def __init__(self, data, children=(), op=''):
        # Numerical value stored in this node
        self.data = data

        # Gradient of the final output with respect to this node
        self.grad = 0.0

        # Function that will propagate gradients to parent nodes
        self._backward = lambda: None

        # Parent nodes in the computational graph
        self._prev = set(children)

        # Operation label (for debugging/visualization)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


    # -----------------------------
    # BASIC OPERATORS
    # -----------------------------

    def __add__(self, other):
        # Wrap raw numbers into Value objects
        other = other if isinstance(other, Value) else Value(other)

        # Create output node
        out = Value(self.data + other.data, (self, other), '+')

        # Backward rule for addition: derivative is 1 for both inputs
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out


    def __mul__(self, other):
        # Wrap raw numbers into Value objects
        other = other if isinstance(other, Value) else Value(other)

        # Create output node
        out = Value(self.data * other.data, (self, other), '*')

        # Backward rule for multiplication:
        # d(x*y)/dx = y, d(x*y)/dy = x
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out


    # -----------------------------
    # UNARY AND BINARY OPERATORS
    # -----------------------------

    def __neg__(self):
        # Unary negation: -x
        return self * -1

    def __sub__(self, other):
        # Subtraction: x - y = x + (-y)
        return self + (-other)

    def __radd__(self, other):
        # Right-hand addition: allows number + Value
        return self + other

    def __rmul__(self, other):
        # Right-hand multiplication: allows number * Value
        return self * other

    def __rsub__(self, other):
        # Right-hand subtraction: number - Value
        return other + (-self)


    # -----------------------------
    # POWER OPERATOR
    # -----------------------------

    def __pow__(self, n):
        # Power: x**n
        out = Value(self.data ** n, (self,), f'**{n}')

        # Backward rule: d(x**n)/dx = n * x**(n-1)
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad

        out._backward = _backward
        return out


    # -----------------------------
    # DIVISION
    # -----------------------------

    def __truediv__(self, other):
        # Division: x / y = x * (y**-1)
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)


    # -----------------------------
    # EXPONENTIAL
    # -----------------------------

    def exp(self):
        import math
        e = math.exp(self.data)

        out = Value(e, (self,), 'exp')

        # Backward rule: d(exp(x))/dx = exp(x)
        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out


    # -----------------------------
    # LOGARITHM
    # -----------------------------

    def log(self):
        import math
        out = Value(math.log(self.data), (self,), 'log')

        # Backward rule: d(log(x))/dx = 1/x
        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out


    # -----------------------------
    # TANH ACTIVATION
    # -----------------------------

    def relu(self):
        # ReLU activation: max(0, x)
        out = Value(max(0, self.data), (self,), 'relu')

        # Backward rule: gradient flows only if output > 0
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        import math
        t = math.tanh(self.data)

        out = Value(t, (self,), 'tanh')

        # Backward rule: d(tanh(x))/dx = 1 - tanh(x)^2
        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
      import math
      s = 1 / (1 + math.exp(-self.data))
      out = Value(s, (self,), 'sigmoid')

      def _backward():
          self.grad += (s * (1 - s)) * out.grad

      out._backward = _backward
      return out

    def leaky_relu(self, alpha=0.01):
      out = Value(self.data if self.data > 0 else alpha * self.data, (self,), 'leaky_relu')

      def _backward():
          self.grad += (1 if self.data > 0 else alpha) * out.grad

      out._backward = _backward
      return out

    def softplus(self):
      import math
      sp = math.log(1 + math.exp(self.data))
      out = Value(sp, (self,), 'softplus')

      def _backward():
          self.grad += (1 / (1 + math.exp(-self.data))) * out.grad

      out._backward = _backward
      return out



    # -----------------------------
    # BACKWARD PASS
    # -----------------------------

    def backward(self):
        # Build topological order of nodes
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Seed gradient of final output (dL/dL = 1)
        self.grad = 1.0

        # Traverse nodes in reverse topological order
        for v in reversed(topo):
            v._backward()
