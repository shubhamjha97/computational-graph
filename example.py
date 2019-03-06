import computational_graph as cg
from computational_graph import Var, gradient


# Addition
x = Var(4) 			# create new variable
y = Var(5) 			# create new variable

z = x + y 			# add two variables
z1 = z + 10 		# add variable with constant

add_fn = cg.add()
z2 = add_fn(x, y) 	# another way to use the api

print(z)
print(z1)
print(z2)

# Multiplication
z3 = z * 5
print(z3)

print(gradient(z3, x))

# Exponentiation
x = Var(2)
y = Var(5)

z = x ** y

print(z)
print(gradient(z, x))
print(gradient(z, y))
