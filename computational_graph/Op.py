from .basic import Var
import math


class add:
	def __init__(self):
		self.a = None
		self.b = None
		self.parent1 = None
		self.parent2 = None

	def __call__(self, a, b):
		self.a = a if type(a) is Var else Var(a)
		self.b = b if type(b) is Var else Var(b)
		x = Var(self.a.val + self.b.val)

		# modify gradients too
		self.parent1 = self.a
		self.parent2 = self.b
		self.child = x
		x.parent = self
		return x

	def __back__(self, upstream_grad):
		self.parent1.__back__(upstream_grad * 1)
		self.parent2.__back__(upstream_grad * 1)


class mul:
	def __init__(self):
		self.a = None
		self.b = None
		self.parent1 = None
		self.parent2 = None

	def __call__(self, a, b):
		self.a = a if type(a) is Var else Var(a)
		self.b = b if type(b) is Var else Var(b)
		x = Var(self.a.val * self.b.val)

		# modify gradients too
		self.parent1 = self.a
		self.parent2 = self.b
		self.child = x
		x.parent = self
		return x

	def __back__(self, upstream_grad):
		self.parent1.__back__(upstream_grad * self.parent2.val)
		self.parent2.__back__(upstream_grad * self.parent1.val)


class pow:
	def __init__(self):
		self.base = None
		self.exponent = None
		self.parent_base = None
		self.parent_exponent = None

	def __call__(self, base, exponent):
		self.base = base if type(base) is Var else Var(base)
		self.exponent = exponent if type(exponent) is Var else Var(exponent)
		x = Var(self.base.val ** self.exponent.val)

		self.parent_base = self.base
		self.parent_exponent = self.exponent
		self.child = x
		x.parent = self
		return x

	def __back__(self, upstream_grad):
		self.parent_base.__back__(upstream_grad * self.exponent.val * (self.base.val) ** (self.exponent.val - 1))
		self.parent_exponent.__back__(upstream_grad * self.child.val * math.log(self.base.val))


def backprop(var):
	var.parent.__back__(1)


def gradient(y, x):
	'''
	compute dy/dx
	'''
	assert type(x) is Var
	assert type(y) is Var

	assert x.requires_grad is True
	assert y.requires_grad is True

	y.upstream_grad = 1
	y.grad = 1

	backprop(y)

	res = x.grad

	return res
