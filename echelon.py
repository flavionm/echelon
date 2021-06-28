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
from sympy import I, MutableMatrix, div, expand, oo, symbols
from sympy.polys import polytools
from sympy.polys.polyerrors import ComputationFailed


def _main():
    z = symbols('z')

    # Insert your matrix here, in fractional form on QQ_I
    polynomial_matrix = MutableMatrix([[27*z/10, 27*z**2, 5*z**-7, z**-4],
                                       [0, z**2, z**7, z**-1 + z**2 + 1],
                                       [0, 0, z**-7, z**-5],
                                       [0, 0, 0, 1 + I]])

    dedekind_weber(polynomial_matrix, z)


def dedekind_weber(polynomial_matrix, variable):
    """Executes the Dedekind-Weber proof's algorithm in Q+IQ Laurentz polynomials"""

    min_degree = _minimun_degree(polynomial_matrix, variable)

    positive_matrix = _regulatization(polynomial_matrix, variable, min_degree)

    if not _is_invertible(positive_matrix, variable, min_degree):
        raise ValueError('Invalid matrix, must be invertible')

    positive_matrix, min_degree = _echelon(
        positive_matrix, 0, variable, min_degree)

    _print_result(positive_matrix, min_degree)


def _is_invertible(M, z, min_degree):
    determinant = expand(M.det())
    print('Determinant is', expand(z**(len(M[:, 0])*min_degree)*determinant))
    leading_monomials = expand(polytools.LT(determinant))
    return determinant != 0 and determinant == leading_monomials


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

    M, min_degree = _echelon(M, level + 1, z, min_degree)

    while polytools.degree(M[level, level], z) < polytools.degree(M[level + 1, level + 1], z):
        for i in range(level + 1, len(M[:, 0])):
            all_terms = expand(M[i, level]).as_poly(z).all_terms()
            remainder = sum(z**n * term for (n,), term in all_terms if n <=
                            polytools.degree(M[level, level], z))
            correction = remainder / M[level, level]
            for j in range(0, len(M[:, 0])):
                M[i, j] -= correction * M[level, j]
            new_min_degree = _minimun_degree(expand(M), z)
            M = MutableMatrix(_regulatization(expand(M), z, new_min_degree))
            min_degree += new_min_degree

        second_line = M[level + 1, :]
        M[level + 1, :] = M[level, :]
        M[level, :] = second_line

        M[level:, level:] = _echelon_first_line(M[level:, level:], z)

        M, min_degree = _echelon(M, level + 1, z, min_degree)

    return MutableMatrix(expand(M)), min_degree


def _echelon_first_line(M, z):
    while True:
        done = True
        for entry in M[0, 1:]:
            if entry != 0:
                done = False
                break

        if done:
            return M

        lower_degree = oo
        for entry in M[0, :]:
            all_terms = expand(entry).as_poly(z).all_terms()
            new_degree = min(n for (n,), _ in all_terms)
            if new_degree < lower_degree:
                lower_degree = new_degree

        coeffs = []
        for entry in M[0, :]:
            coeffs.append(expand(entry / z**lower_degree))

        for i in range(len(M[0, :])):
            for j in range(i, 0, -1):
                degree_current = polytools.degree(coeffs[j], z)
                if coeffs[j] == 0:
                    degree_current = oo
                degree_before = polytools.degree(coeffs[j - 1], z)
                if coeffs[j - 1] == 0:
                    degree_before = oo
                if degree_current <= degree_before:
                    second_column = M[:, j]
                    M[:, j] = M[:, j - 1]
                    M[:, j - 1] = second_column

                    coeffs[j], coeffs[j - 1] = coeffs[j - 1], coeffs[j]

        for i in range(1, len(coeffs)):
            quotient, _ = div(coeffs[i], coeffs[0], z)
            for j in range(len(coeffs)):
                factor = quotient * z**lower_degree * M[j, 0]
                M[j, i] = expand(M[j, i] - factor)


def _print_result(positive_matrix, min_degree):
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
