from __future__ import print_function
import math
import sys

"""
Given epsilon, delta, and dvc, this calculates the minimum N satisfying the inequality

(let e be epsilon)
	N >= (8/e^2) ln(4((2N)^dvc + 1) / delta)

This is just a simple search procedure. See AML book, Mostafa p. 
"""


def _getMillerRhs(N, epsilon, delta, dvc):
	return	(8.0 / epsilon**2) * math.log( (4 * (2*N)**dvc + 1) / delta)

def _getBookRhs(N, epsilon, delta, dvc):
	return	(8.0 / epsilon**2) * math.log(4 * ((2*N + 1)**dvc) / delta)


def _f(n, epsilon, delta, dvc):
	return n - (8 / epsilon**2.0) * (dvc * math.log(2*n) +math.log(4) - math.log(delta))

"""
Implements the first derivative of the millerRhs equation above, but without the +1 in the integral
"""
def _fprime(n, epsilon, dvc):
	return 1.0 - (4.0 * dvc) / (epsilon**2.0 * n)

def _fDoublePrime(n, epsilon, dvc):
	return (-4.0 * dvc) / (epsilon**2.0 * n**2.0)

"""
Finds roots
"""
def NewtonRaphson(epsilon, delta, dvc):
	n_t1 = 2
	n_t0 = 3000000000

	while True:
		n_t1 = n_t0 - _f(n_t0, epsilon, delta, dvc) / _fprime(n_t0, epsilon, dvc)
		print("Current estimate: "+str(n_t1))
		n_t0 = n_t1

def Halleys(epsilon, delta, dvc):
	n_t1 = 1
	n_t0 = 300000000

	while True:
		num = 2.0 * _f(n_t0, epsilon, delta, dvc) * _fprime(n_t0, epsilon, dvc)
		den = 2.0 * (_fprime(n_t0, epsilon, dvc))**2.0 - _f(n_t0, epsilon, delta, dvc) * _fDoublePrime(n_t0, epsilon, dvc)
		n_t1 = n_t0 - num / den
		print("Current estimate: "+str(n_t1))
		n_t0 = n_t1

"""
Given epsilon (e), delta (d), and dvc, this implements a simple search for the N satisfying the
inequality in the header.
"""
def BruteSearchN(epsilon,delta,dvc):
	N = 0
	rhs = 1

	while N < rhs:
		N += 1
		rhs = _getMillerRhs(N, epsilon, delta, dvc)

	return N

"""
Finds N in a more analytic fashion than BruteSearchN.

No good, gives incorrect result
"""
def Principled(epsilon, dvc):
	return (8 * dvc) / (epsilon**2) + 1


def usage():
	print("usage: python ./ndvc.py [epsilon] [delta] [dvc]")


if len(sys.argv) != 4:
	print("wrong numn args")
	usage()
	exit()

epsilon = float(sys.argv[1])
delta = float(sys.argv[2])
dvc = float(sys.argv[3])

N = BruteSearchN(epsilon, delta, dvc)
print("N: "+str(N))
N = Principled(epsilon, dvc)
print("N: "+str(N))

#Halleys(epsilon, delta, dvc)
NewtonRaphson(epsilon, delta, dvc)


