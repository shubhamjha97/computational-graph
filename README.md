# computational-graph

A simple library that builds computational graphs and provides gradient computation capabilities. Currently supports only simple datatypes such as int and float.

## Currently supported ops
- Addition
- Subtraction
- Multiplication
- Division
- Exponentiation

## Other feratures-
- Automatically casts ints, floars to Var
- `gradient` function provides autodiff capabilities

## Examples

### Sigmoid

# Future work
- extend to numpy arrays
- add functionality for plotting the computational graph
- add support for detaching gradients
- add support for commonly used neural network functions (e.g. sigmoid, softmax, ReLU, etc.)