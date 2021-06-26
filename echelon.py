"""Dedeking-Weber algorithm"""
from sympy import Matrix, gcdex, symbols, simplify
from sympy.polys import polytools


def _main():
    z = symbols('z')

    polynomial_matrix = Matrix([[z, z**2, 0],
                                [z**-1, z**3 + 1, 5*z**27],
                                [0, 0, z]])

    dedekind_weber(polynomial_matrix, z)


def dedekind_weber(polynomial_matrix, variable):
    """Executes the Dedekind-Weber proof's algorithm in complex Laurentz polynomials"""
    if not _is_invertible(polynomial_matrix):
        raise ValueError('Invalid matrix, must be invertible')

    min_degree = _minimun_degree(polynomial_matrix, variable)

    positive_matrix = _regulatization(polynomial_matrix, variable, min_degree)

    _echelon(positive_matrix)

    print(positive_matrix)


def _is_invertible(M):
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


def _echelon(M):
    if len(M) == 1:
        return M

    _echelon_first_line(M)

    M[1:, 1:] = _echelon(M[1:, 1:])

    return M


def _echelon_first_line(M):
    bezout_coefficients, gcd = _calculate_bezout_coefficients(M[0, :])

    for i, _ in enumerate(M[:, 0]):
        total = 0
        for j, entry in enumerate(M[i, :]):
            total += bezout_coefficients[j] * entry
        M[i, 0] = simplify(total)

    for i in range(1, len(M[0, :])):
        quotient = M[0, i] / gcd
        for j, entry in enumerate(M[:, i]):
            M[j, i] = simplify(entry - quotient * M[j, 0])


def _calculate_bezout_coefficients(top_line):
    if len(top_line) == 1:
        return [1], top_line[0]
    if len(top_line) == 2:
        if top_line[0] == 0:
            return [1, 1], top_line[1]
        if top_line[1] == 0:
            return [1, 1], top_line[0]

        coef1, coef2, gcd = gcdex(top_line[0], top_line[1])
        return [coef1, coef2], gcd

    previous_coefficients, previous_gcd = _calculate_bezout_coefficients(
        top_line[:-1])

    coef1, coef2, gcd = 1, 1, previous_gcd
    if top_line[-1] != 0:
        coef1, coef2, gcd = gcdex(previous_gcd, top_line[-1])

    coefficients = []
    for coef in previous_coefficients:
        coefficients.append(coef * coef1)

    coefficients.append(coef2)
    return coefficients, gcd


if __name__ == '__main__':
    _main()
