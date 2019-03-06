import computational_graph as cg


class Op:
	def __init__(self):
		self.parents = None
		self.child = None
		self.shape = None
		self.val = None
		self.upstream_grad = None
		self.grad = None

	def __call__(self):
		pass

	def __back__(self, upstream_grad):
		pass


class Var:
	def __init__(self, val=None, requires_grad=True):
		self.val = val
		self.dtype = type(self.val)
		self.parent = None
		self.requires_grad = requires_grad

	def __add__(self, other):
		if type(other) is not Var: other = Var(other)
		fn = cg.add()
		res = fn(self, other)
		return res

	def __mul__(self, other):
		if type(other) is not Var: other = Var(other)
		fn = cg.mul()
		res = fn(self, other)
		return res

	def __pow__(self, other):
		if type(other) is not Var: other = Var(other)
		fn = cg.pow()
		res = fn(self, other)
		return res

	def __str__(self):
		return "{}".format(self.val)

	def __repr__(self):
		return "cg_variable({}, dtype={})".format(self.val, self.dtype)

	def __back__(self, upstream_grad):
		self.grad = upstream_grad
		if self.parent is not None:
			self.parent.__back__(upstream_grad)
