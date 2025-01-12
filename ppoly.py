from __future__ import annotations
from typing import *

import scipy.interpolate
import bisect
import heapq
import math
import numpy
import itertools

class PPoly(scipy.interpolate.PPoly):
    def __init__(self, x, c):
        x = numpy.array(x, dtype="object").flatten()
        if (len(x) > 1) and (x[0] > x[1]):
            raise ValueError("`x` must be strictly increasing.")
        while len(c) > 1 and all([cc == 0 for cc in c[0]]):
            c = c[1:]
        super().__init__(c, x)

    @staticmethod
    def __addPoly(c1, c2):
        if len(c1) == len(c2):
            return c1 + c2
        elif len(c1) < len(c2):
            return PPoly.__addPoly(c2, c1)
        else: # len(c1) > len(c2)
            return c1 + numpy.append(numpy.zeros((len(c1)-len(c2), 1)), c2, axis=0)

    @staticmethod
    def __mulPoly(c1, c2):
        newc = numpy.zeros((len(c1) + len(c2) - 1, 1))
        for (i, v1) in enumerate(reversed(c1)):
            for (j, v2) in enumerate(reversed(c2)):
                newc[len(newc) - 1 - (i + j)][0] += v1[0] * v2[0]
        return newc

    @staticmethod
    def __insertPoly(x, outer, otherpows):
        result = []
        for (i, coef) in enumerate(reversed(outer[otherpows[1](x)].c)):
            relevantPow = otherpows[i][x]
            #result = numpy.array(list(itertools.chain(itertools.repeat([0], len(relevantPow.c) + i - len(result)), result))) + numpy.array(list(itertools.chain(coef * relevantPow.c, itertools.repeat([0], i))))
            result = numpy.array(list(itertools.chain(itertools.repeat([0], len(relevantPow.c) - len(result)), result))) + coef * relevantPow.c
        return result

    @staticmethod
    def __insertPolyOld(x, outer, otherpows):
        result = []
        for (i, coef) in enumerate(reversed(outer[x].c)):
            relevantPow = otherpows[i][x]
            #result = numpy.array(list(itertools.chain(itertools.repeat([0], len(relevantPow.c) + i - len(result)), result))) + numpy.array(list(itertools.chain(coef * relevantPow.c, itertools.repeat([0], i))))
            result = numpy.array(list(itertools.chain(itertools.repeat([0], len(relevantPow.c) - len(result)), result))) + coef * relevantPow.c
        return result

    def __neg__(self):
        return PPoly(self.x, -self.c)

    def __add__(self, other) -> PPoly:
        if isinstance(other, PPoly):
            newx = []
            newc = [[]] * max(len(self.c), len(other.c))
            lastx = None
            lastpoly = []
            for x in list(heapq.merge(self.x, other.x))[0:-1]:
                if x != lastx:
                    newpoly = PPoly.__addPoly(self[x].c, other[x].c)
                    if len(newpoly) != len(lastpoly) or any(newpoly != lastpoly):
                        lastpoly = newpoly
                        newx.append(x)
                        newc = [c1 + c2 for (c1, c2) in zip(newc, [[0]]*(len(newc) - len(newpoly)) + newpoly.tolist())]
                        lastx = x
            return PPoly(newx + [max(self.x[-1], other.x[-1])], newc)
        else:
            return self + PPoly([self.x[0], self.x[-1]], numpy.array(other, ndmin=2))

    def __sub__(self, other) -> PPoly:
        return self + -other

    def __mul__(self, other) -> PPoly:
        if isinstance(other, PPoly):
            newx = []
            newc = [[]] * (len(self.c) + len(other.c))
            lastx = None
            lastpoly = []
            for x in list(heapq.merge(self.x, other.x))[0:-1]:
                if x != lastx:
                    newpoly = PPoly.__mulPoly(self[x].c, other[x].c)
                    if len(newpoly) != len(lastpoly) or any(newpoly != lastpoly):
                        lastpoly = newpoly
                        newx.append(x)
                        newc = [c1 + c2 for (c1, c2) in zip(newc, [[0]]*(len(newc) - len(newpoly)) + newpoly.tolist())]
                        lastx = x
            return PPoly(newx + [max(self.x[-1], other.x[-1])], newc)
        else:
            return PPoly(self.x, self.c * other)

    def __rmul__(self, other):
        return self * other

    @staticmethod
    def get_intersection_range(x1, x2):
        minx = max(x1[0], x2[0])
        maxx = min(x1[-1], x2[-1])
        return numpy.unique([x for x in itertools.chain(x1, x2) if x >= minx and x <= maxx])

    def __truediv__(self, other) -> PPoly:
        if isinstance(other, PPoly):
            if other.c.shape[0] != 1:
                raise ValueError('Divisor of PPoly division must be a piecewise constant function.')
            newx = PPoly.get_intersection_range(self.x, other.x)
            newc = [[]] * len(self.c)
            lastpoly = []
            for x in newx[0:-1]:
                newpoly = self[x].c / other[x].c[0][0]
                if len(newpoly) != len(lastpoly) or any(newpoly != lastpoly):
                    lastpoly = newpoly
                    newc = [c1 + c2 for (c1, c2) in zip(newc, [[0]]*(len(newc) - len(newpoly)) + newpoly.tolist())]
            return PPoly(newx, newc)
        else:
            return PPoly(self.x, self.c / other)

    def __str__(self) -> str:
        result = ""
        x = ["{:.2f}".format(val) for val in self.x]
        maxlen = max([len(val) for val in x])
        for (i, val) in enumerate(self.c.T):
            result += " "*(maxlen-len(x[i]))
            result += x[i]
            result += " - "
            result += " "*(maxlen-len(x[i+1]))
            result += x[i+1]
            result += ": "
            for (i, c) in enumerate(val):
                result += "{}".format(c)
                result += "*x^^("
                result += "{}".format(len(val) - 1 - i)
                result += ") + "
            result = result[:-3]
            result += "\n"
        return result

    def __call__(self, other, nu=0):
        if isinstance(other, PPoly):
            otherpows = [PPoly([-math.inf, math.inf], [[1]]), other]
            for _ in range(len(self.c) - 1):
                otherpows.append(otherpows[-1] * other)

            # get relevant segment change positions of outer
            outerchanges = []
            for x in self.x:
                changes = other.solve(x) # might only work if crossed increasingly -> only for monotonically increasing functions
                changes = [c for c in sorted(changes) if not math.isnan(c) and abs(other(c) - x) < 0.0001] # bug in scipy regarding solve, please fix

                # sanity check
                #if len(changes) >= 2:
                #    for c in changes[0:-1]:
                #        if c >= 0:
                #            raise RuntimeError('Inner function is not monotonically increasing.')

                if len(changes) > 0:
                    outerchanges = outerchanges + [changes[-1]]

            newx = []
            newc = [[]] * (len(self.c) * len(other.c)) # todo: check bounds
            lastx = None
            lastpoly = []
            for x in list(heapq.merge(outerchanges, other.x))[0:-1]:
                if x != lastx:
                    newpoly = PPoly.__insertPoly(x, self, otherpows)
                    if len(newpoly) != len(lastpoly) or any(newpoly != lastpoly):
                        lastpoly = newpoly
                        newx.append(x)
                        newc = [c1 + c2 for (c1, c2) in zip(newc, [[0]]*(len(newc) - len(newpoly)) + newpoly.tolist())]
                        lastx = x
            return PPoly(newx + [max(outerchanges[-1], other.x[-1])], newc)

            # wrong first implementation
            if len(self.c) == 1:
                return self

            otherpows = [PPoly([-math.inf, math.inf], [[1]]), other]
            for _ in range(len(self.c) - 1):
                otherpows.append(otherpows[-1] * other)

            newx = []
            newc = [[]] * ((len(self.c) - 1) * len(other.c)) # todo: check bounds
            lastx = None
            lastpoly = []
            for x in list(heapq.merge(self.x, other.x))[0:-1]:
                if x != lastx:
                    newpoly = PPoly.__insertPolyOld(x, self, otherpows)
                    if len(newpoly) != len(lastpoly) or any(newpoly != lastpoly):
                        lastpoly = newpoly
                        newx.append(x)
                        newc = [c1 + c2 for (c1, c2) in zip(newc, [[0]]*(len(newc) - len(newpoly)) + newpoly.tolist())]
                        lastx = x
            return PPoly(newx + [max(self.x[-1], other.x[-1])], newc)
        else:
            return super().__call__(other, nu)

    def roots(self, discontinuity=True):
        result = list(super().roots(False))
        if discontinuity:
            lastsector = self[self.x[0]]
            for bp in self.x[1:-1]:
                thissector = self[bp]
                before = lastsector(bp)
                after = thissector(bp)
                if (before < 0 and after > 0) or (before > 0 and after < 0):
                    result.append(bp)
                lastsector = thissector
        return result

    def __getitem__(self, x) -> PPoly:
        if isinstance(x, slice):
            if x.step is not None:
                raise ValueError('Slices with steps not supported.')
            startpos = x.start if x.start is not None else self.x[0]
            endpos = x.stop if x.stop is not None else self.x[-1]
            startidx = bisect.bisect(self.x, startpos)
            startidx = startidx - 1 if startidx > 0 else 0
            endidx = bisect.bisect(self.x, endpos)
            endidx = endidx - 1 if endidx > 0 else 0
            newx = list(self.x[startidx:endidx+1])
            newx[0] = startpos
            if endpos > self.x[-1] and len(newx) > 1:
                newx[-1] = endpos
            elif endpos not in self.x:
                newx.append(endpos)
            startidx = startidx if startidx < len(self.c[0]) else len(self.c[0])-1
            endidx = endidx if endidx < len(self.c[0]) else len(self.c[0])-1
            endidx = endidx-1 if endpos in self.x[:-1] else endidx
            return PPoly(newx, [e[startidx:endidx+1] for e in self.c])
        else:
            idx = bisect.bisect(self.x, x)
            idx = idx - 1 if idx > 0 else 0
            startpos = self.x[len(self.x) - 2] if idx >= len(self.x) - 1 else self.x[idx] if idx > 0 else -math.inf
            endpos = self.x[idx + 1] if idx + 1 < len(self.x) else math.inf
            return PPoly([startpos, endpos], [[val[idx if idx < len(val) else len(val)-1]] for val in self.c])

    @staticmethod
    def _equal_coefficients(a: numpy.ndarray, b: numpy.ndarray) -> bool:
        if len(a) == len(b):
            return all(a == b)
        elif len(a) < len(b):
            a, b = b, a # a is always longer than b
        return all(a[0:len(a) - len(b)] == numpy.zeros(len(a) - len(b))) and all(a[len(a) - len(b):] == b)

    def __setitem__(self, x: slice, val: PPoly):
        startpos = x.start if x.start is not None else -math.inf
        endpos = x.stop if x.stop is not None else math.inf

        if startpos > self.x[0]:
            result = self[:startpos]
            insert_ppoly = val[startpos:endpos]
            if not self._equal_coefficients(result.c.T[-1], insert_ppoly.c.T[0]):
                result.extend(insert_ppoly.c, insert_ppoly.x[1:])
            else:
                result.x[-1] = insert_ppoly.x[1]
                if len(insert_ppoly.x) > 2:
                    insert_ppoly = PPoly(insert_ppoly.x[1:], insert_ppoly.c.T[1:].T)
                    result.extend(insert_ppoly.c, insert_ppoly.x[1:])
        else:
            result = val[startpos:endpos]

        if self.x[-1] > endpos:
            insert_ppoly = self[endpos:]
            if not self._equal_coefficients(result.c.T[-1], insert_ppoly.c.T[0]):
                result.extend(insert_ppoly.c, insert_ppoly.x[1:])
            else:
                result.x[-1] = insert_ppoly.x[1]
                if len(insert_ppoly.x) > 2:
                    insert_ppoly = PPoly(insert_ppoly.x[1:], insert_ppoly.c.T[1:].T)
                    result.extend(insert_ppoly.c, insert_ppoly.x[1:])

        self.x = result.x
        self.c = result.c

    def get_next_change(self, x: float) -> float:
        pos = bisect.bisect(self.x, x) # numpy.searchsorted
        pos = pos if pos != 0 and pos != len(self.x)-1 else pos + 1
        return self.x[pos] if pos < len(self.x) else math.inf

    #def get_plot_points(self, start: float, end: float, step: float = 0) -> Tuple[numpy.ndarray, numpy.ndarray]:
    #    if step == 0:
    #        step = (end - start) / 1000
    #    number_of_points = (end - start) / step
    #    x = numpy.linspace(start, end, number_of_points)
    #    y = self(x)
    #    return x, y


if __name__ == "__main__":
    p1 = PPoly([-10, 2, 100], [[2, 2], [1, 1], [-1, -4]])
    p2 = PPoly([4, 8, 12, 17], [[-10, 2, 3], [1, 2, 3], [4, 5, 6]])
    c = PPoly([-5, -2, 3, 10], [[5, 0, 1]])
    #print(p1 + p2)
    #print(p1 * p2)
    #print(p1(p2))
    #print(p2 / c)
    #print(p1(-math.inf), p2(-math.inf), c(-math.inf))
    #print(p1[:-20])
    #p2[20:50] = p1
    #print(p2)
    p1[6:50] = p2
    print(p1)
