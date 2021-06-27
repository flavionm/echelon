# Copyright 2021 - Flávio Sousa - Thadeu Henrique Cardoso

# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Dedeking-Weber algorithm"""
from sympy import I, MutableMatrix, div, gcdex, simplify, symbols, expand
from sympy.polys import polytools
from sympy.polys.polyerrors import ComputationFailed


def _main():
    z = symbols('z')

    # Insert your matrix here, in fractional form on QQ_I
    # polynomial_matrix = MutableMatrix([[I*z - 1, 27*z**2/10, 5*z**-7, z**-4],
    #                             [z**-5, 0, z**7/(1+I), z**-1 + z**2 + 1],
    #                             [0, 0, z**-15, z**-5],
    #                             [0, 0, 0, z**2]])

    polynomial_matrix = MutableMatrix([[z - 1, 27*z**2, 5*z**-7, z**-4],
                                [z**-5, 0, z**7, z**-1 + z**2 + 1],
                                [0, 0, z**-15, z**-5],
                                [0, 0, 0, z**2]])

    # polynomial_matrix = Matrix([[2*I*z, 1, 1, 1],
                                # [0, 1, 1, 1],
                                # [0, 0, z**-15, 1],
                                # [0, 0, 0, z**2]])

    dedekind_weber(polynomial_matrix, z)

    # print(((z + z**2)/(z**3)).as_poly(z, 1/z))

    # all_terms = (z**3 + 2*z**2 + 3*z).as_poly().all_terms()
    # print(sum(z**n * term for (n,), term in all_terms if n <= 2))

def dedekind_weber(polynomial_matrix, variable):
    """Executes the Dedekind-Weber proof's algorithm in Q+IQ Laurentz polynomials"""
    if not _is_invertible(polynomial_matrix):
        raise ValueError('Invalid matrix, must be invertible')

    min_degree = _minimun_degree(polynomial_matrix, variable)

    positive_matrix = _regulatization(polynomial_matrix, variable, min_degree)

    positive_matrix, min_degree = _echelon(positive_matrix, 0, variable, min_degree)

    _print_result(positive_matrix, min_degree)


def _is_invertible(M):
    determinant = M.det()
    leading_monomials = polytools.LT(determinant)
    return determinant == leading_monomials


def _minimun_degree(M, z):
    min_degree = 0
    for p in M:
        degree = polytools.degree(p, 1/z)
        if degree > min_degree:
            min_degree = degree
    return -min_degree


def _regulatization(M, z, min_degree):
    return z**-min_degree*M


def _echelon(M, level, z, min_degree):
    if level == len(M[:, 0]) - 1:
        return M, min_degree

    M[level:, level:] = _echelon_first_line(M[level:, level:], z)

    # print('before', level, M)

    M, min_degree = _echelon(M, level + 1, z, min_degree)

    # print('current', level, M)

    # print(polytools.degree(M[level, level], z), polytools.degree(M[level + 1, level + 1], z))
    # print(polytools.degree(M[level, level], z) < polytools.degree(M[level + 1, level + 1], z))
    while polytools.degree(M[level, level], z) < polytools.degree(M[level + 1, level + 1], z):
        # print('before doing', level, M)
        for i in range(level + 1, len(M[:, 0])):
            print('matrix', expand(M))
            print('termos', expand(M[i, level]), ',', M[level, level])
            all_terms = expand(M[i, level]).as_poly(z).all_terms()
            remainder = sum(z**n * term for (n,), term in all_terms if n <= polytools.degree(M[level, level], z))
            correction = remainder / M[level, level]
            print('correction', correction)
            for j in range(0, len(M[:, 0])):
                M[i, j] -= correction * M[level, j]
            new_min_degree = _minimun_degree(expand(M), z)
            M = MutableMatrix(_regulatization(expand(M), z, new_min_degree))
            min_degree += new_min_degree

        second_line = M[level + 1, :]
        M[level + 1, :] = M[level, :]
        M[level, :] = second_line

        new_min_degree = _minimun_degree(expand(M), z)
        M = MutableMatrix(_regulatization(expand(M), z, new_min_degree))
        min_degree += new_min_degree

        # print('doing', level, M)

        M[level:, level:] = _echelon_first_line(M[level:, level:], z)

        M, min_degree = _echelon(M, level + 1, z, min_degree)
        # print('after doing', level, M)
        # print(polytools.degree(M[level, level], z), polytools.degree(M[level + 1, level + 1], z))
        # print(polytools.degree(M[level, level], z) < polytools.degree(M[level + 1, level + 1], z))

    # print('after', level, M)
    return MutableMatrix(expand(M)), min_degree


def _echelon_first_line(M, z):
    bezout_coefficients, gcd = _calculate_bezout_coefficients(M[0, :])

    if bezout_coefficients[0] == 0:
        for i, coeff in enumerate(bezout_coefficients):
            if coeff != 0:
                second_column = M[:, i]
                M[:, i] = M[:, 0]
                M[:, 0] = second_column

                bezout_coefficients[0], bezout_coefficients[i] = bezout_coefficients[i], bezout_coefficients[0]
                break

    for i, _ in enumerate(M[:, 0]):
        total = 0
        for j, entry in enumerate(M[i, :]):
            total += bezout_coefficients[j] * entry
        M[i, 0] = expand(total)

    for i in range(1, len(M[0, :])):
        quotient, _ = div(M[0, i], gcd, z)
        for j, entry in enumerate(M[:, i]):
            M[j, i] = expand(entry - quotient * M[j, 0])

    return M


def _calculate_bezout_coefficients(top_line):
    if len(top_line) == 1:
        return [1], top_line[0]
    if len(top_line) == 2:
        if top_line[0] == 0:
            return [1, 1], top_line[1]
        if top_line[1] == 0:
            return [1, 1], top_line[0]

        # print(expand(top_line[0]), expand(top_line[1]))
        coef1, coef2, gcd = gcdex(expand(top_line[0]), expand(top_line[1]))
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


def _print_result(positive_matrix, min_degree):
    print(0)

    output = 'The Dedeking-Weber form is diag('
    for i in range(len(positive_matrix[:, 0])):
        degree = 0
        try:
            degree = polytools.degree_list(positive_matrix[i, i])[0]
        except ComputationFailed:
            pass

        output += 'z^' + \
            str(degree + min_degree) + ','
    print(output[:-1] + ')')


if __name__ == '__main__':
    _main()
