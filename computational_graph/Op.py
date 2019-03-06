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

		self.parent1 = self.a
		self.parent2 = self.b
		self.child = x
		x.parent = self
		return x

	def __back__(self, upstream_grad):
		self.parent1.__back__(upstream_grad * 1)
		self.parent2.__back__(upstream_grad * 1)


class sub:
	'''
	minuend - subtrahend
	'''
	def __init__(self):
		self.minuend = None
		self.subtrahend = None
		self.parent_minuend = None
		self.parent_subtrahend = None

	def __call__(self, minuend, subtrahend):
		self.minuend = minuend if type(minuend) is Var else Var(minuend)
		self.subtrahend = subtrahend if type(subtrahend) is Var else Var(subtrahend)
		x = Var(self.minuend.val - self.subtrahend.val)

		self.parent_minuend = self.minuend
		self.parent_subtrahend = self.subtrahend
		self.child = x
		x.parent = self
		return x

	def __back__(self, upstream_grad):
		self.parent_minuend.__back__(upstream_grad * 1)
		self.parent_subtrahend.__back__(upstream_grad * (-1))


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

		self.parent1 = self.a
		self.parent2 = self.b
		self.child = x
		x.parent = self
		return x

	def __back__(self, upstream_grad):
		self.parent1.__back__(upstream_grad * self.parent2.val)
		self.parent2.__back__(upstream_grad * self.parent1.val)


class div:
	'''
	dividend/divisor
	'''
	def __init__(self):
		self.divident = None
		self.divisor = None
		self.dividend = None
		self.divisor = None

	def __call__(self, dividend, divisor):
		self.dividend = dividend if type(dividend) is Var else Var(dividend)
		self.divisor = divisor if type(divisor) is Var else Var(divisor)
		x = Var(self.dividend.val / self.divisor.val)

		self.parent_dividend = self.dividend
		self.parent_divisor = self.divisor
		self.child = x
		x.parent = self
		return x

	def __back__(self, upstream_grad):
		self.parent_dividend.__back__(upstream_grad / self.parent_divisor.val)
		self.parent_divisor.__back__(upstream_grad * self.parent_dividend.val * (-1) * (1 / (self.parent_divisor.val ** 2)))


class pow:
	'''
	base ** exponent
	'''
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


class exp:
	def __init__(self):
		self.exponent = None
		self.parent_exponent = None

	def __call__(self, exponent):
		self.exponent = exponent if type(exponent) is Var else Var(exponent)
		x = Var(math.exp(self.exponent.val))

		self.parent_exponent = self.exponent
		self.child = x
		x.parent = self
		return x

	def __back__(self, upstream_grad):
		self.parent_exponent.__back__(upstream_grad * math.exp(self.exponent.val))


class neg:
	def __init__(self):
		self.var = None
		self.var = None

	def __call__(self, var):
		self.var = var if type(var) is Var else Var(var)
		x = Var(-1 * self.var.val)

		self.parent_var = self.var
		self.child = x
		x.parent = self
		return x

	def __back__(self, upstream_grad):
		self.parent_var.__back__(upstream_grad * -1)


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
