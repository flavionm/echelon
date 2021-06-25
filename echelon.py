"""Dedeking-Weber algorithm"""
from sympy import symbols
from sympy import Matrix
from sympy.polys import polytools, euclidtools
from sympy import gcdex


def _main():
    z = symbols('z')

    polynomial_matrix = Matrix([[z, z**2],
                                [z**-1, z**3 + 1]])

    dedekind_weber(polynomial_matrix, z)


def dedekind_weber(polynomial_matrix, variable):
    """Executes the Dedekind-Weber proof's algorithm in complex Laurentz polynomials"""
    if not _is_invertible(polynomial_matrix, variable):
        raise ValueError('Invalid matrix, must be invertible')

    min_degree = _minimun_degree(polynomial_matrix, variable)

    positive_matrix = _regulatization(polynomial_matrix, variable, min_degree)

    print(positive_matrix)


def _is_invertible(M, z):
    determinant = M.det()
    leading_monomials = polytools.LT(determinant)
    if determinant == leading_monomials:
        return True
    return False


def _minimun_degree(M, z):
    min_degree = 0
    for p in M:
        degree = polytools.degree(p, 1/z)
        if degree > min_degree:
            min_degree = degree
    return -min_degree


def _regulatization(M, z, min_degree):
    return z**-min_degree*M


if __name__ == '__main__':
    _main()
