from ppoly import PPoly
import math
import itertools

class Func(PPoly):
    def __init__(self, x: list, c: list):
        # extend left to be constant
        #if x[0] != -math.inf:
        #    value = sum(map(lambda a: a[0]*a[1], zip(c[0], map(lambda a: a[0]**a[1], itertools.zip_longest([], range(len(c[0]) - 1, -1, -1), fillvalue=x[0])))))
        #    c = [[0]*(len(c[0])-1) + [value]] + c
        #    x.insert(0, -math.inf)

        # extend right to be constant
        #if x[-1] != math.inf:
        #    value = sum(map(lambda a: a[0]*a[1], zip(c[-1], map(lambda a: a[0]**a[1], itertools.zip_longest([], range(len(c[-1]) - 1, -1, -1), fillvalue=x[-1])))))
        #    c = c + [[0]*(len(c[0]) - 1) + [value]]
        #    x.append(math.inf)

        super().__init__(x, list(map(list, zip(*c))))

        # check for weak monotonic increase
        d1 = self.derivative()
        d2 = d1.derivative()

        for bpx in self.x[:-1]:
            if d1(bpx) < 0:
                raise ArithmeticError('Piecewise defined polynomial must be monotonically increasing.')

        for xp in d1.roots():
            if d2(xp) < 0:
                raise ArithmeticError('Piecewise defined polynomial must be monotonically increasing.')

        #for bs in range(1, len(self.x) - 1):
        #    before = self[self.x[bs - 1]](self.x[bs])
        #    after = self[self.x[bs]](self.x[bs])
        #    if after < before:
        #        raise ArithmeticError('Piecewise defined polynomial must be monotonically increasing.')

if __name__ == "__main__":
    f = Func([-1, 1], [[1, 0]])
    print(f(0), f(-100), f(100)) # 0 -1 1
    f = Func([-10, -5, 1], [[0, 1, -100], [-1, 0, 0]])
