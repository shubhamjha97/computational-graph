import computational_graph as cg
from computational_graph import Var, gradient


# Addition
x = Var(4) 			# create new variable
y = Var(5) 			# create new variable

z = x + y 			# add two variables
z1 = z + 10 		# add variable with constant

add_fn = cg.add()
z2 = add_fn(x, y) 	# another way to use the api

print(z)			# 9
print(z1)			# 19
print(z2)			# 9


# Multiplication
z3 = z * 5
print(z3)			# 45

print(gradient(z3, x)) # 5


# Exponentiation
x = Var(2)
y = Var(5)

z = x ** y

print(z)			# 32
print(gradient(z, x))	# 80
print(gradient(z, y))	# 22.18070977791825

# Sigmoid
def sigmoid(x):
	exp_ = cg.exp()
	return 1 / (1 + exp_(-x))

x = Var(0.5)
z = sigmoid(x)
print(z)				# 1.6065306597126334
print(gradient(z, x))	# -0.6065306597126334
