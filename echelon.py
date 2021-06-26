"""Dedeking-Weber algorithm"""
from sympy import Matrix, gcdex, symbols
from sympy.polys import polytools


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

    _echelon(positive_matrix, variable)


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


def _echelon(M, z):
    bezout_coefficients, gcd = _calculate_bezout_coefficients(M[0, :])

    print(bezout_coefficients)
    print(gcd)


def _calculate_bezout_coefficients(top_line):
    if len(top_line) == 1:
        return [1], top_line[0]
    if len(top_line) == 2:
        coef1, coef2, gcd = gcdex(top_line[0], top_line[1])
        return [coef1, coef2], gcd
    previous_coefficients, previous_gcd = _calculate_bezout_coefficients(
        top_line[:-1])
    coef1, coef2, gcd = gcdex(previous_gcd, top_line[-1])

    coefficients = []
    for coef in previous_coefficients:
        coefficients.append(coef * coef1)

    coefficients.append(coef2)
    return coefficients, gcd


if __name__ == '__main__':
    _main()
