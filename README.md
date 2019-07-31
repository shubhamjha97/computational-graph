# computational-graph

A simple library that builds computational graphs and provides gradient computation capabilities. Currently supports only simple datatypes such as int and float.

Please note that this code has been tested only with Python3.6 and above. The code is not guaranteed to work correctly on lower versions of Python.

## Currently supported operations
- Addition
- Subtraction
- Multiplication
- Division
- Exponentiation

## Other feratures-
- Automatically casts `ints`, `floats` to `Var`
- `gradient` function provides autodiff capabilities

## Examples
Import the library-
```python
>>> import computational_graph as cg
>>> from computational_graph import Var, gradient
```

### Basic ops
```python
x = Var(5)
y = Var(6)
z = x + y	# Addition
print(z) 	# 11

type(z)		# <class 'computational_graph.basic.Var'>

# Also note that, we can also perform an operation with a Var and an int or float-
z = x + 6	# 11
type(z)		# <class 'computational_graph.basic.Var'>

# Other supported basic ops
z = x - y	# Subtraction
z = x * y	# Multiplication
z = x / y	# Division
z = x ** y	# Exponentiation
```

### Other ops
#### Sigmoid
```python
import computational_graph as cg
from computational_graph import Var, gradient


def sigmoid(x):
	exp_ = cg.exp()
	return 1 / (1 + exp_(-x))

x = Var(0.5)
z = sigmoid(x)
print(z)		# 1.6065306597126334
```

### Gradient computation
```python
import computational_graph as cg
from computational_graph import Var, gradient


def sigmoid(x):
	exp_ = cg.exp()
	return 1 / (1 + exp_(-x))

x = Var(0.5)
z = sigmoid(x)
print(z)				# 1.6065306597126334
print(gradient(z, x)) 	# -0.6065306597126334
```


# Future work
- Extend to numpy arrays
- Add functionality for plotting the computational graph
- Add support for detaching gradients
- Add support for commonly used neural network functions (e.g. sigmoid, softmax, ReLU, etc.)
- Detailed documentation (if things work out and I'm able to add numpy arrays support )
