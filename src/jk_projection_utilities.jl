module JKProjection
export generate_fourier_matrices

using SpecialFunctions: loggamma

include("calculate_j_y_eigenstates.jl")
using .JyInJzBasis

function custom_logbinomial(q1, q2)

    if q1 - q2 >= 0
        return loggamma(q1 + 1) - loggamma(q2 + 1) - loggamma(q1 - q2 + 1)
    else
        return -Inf
    end

end

function projection_coeff(L, Qstar, Q1, m)

    return exp(custom_logbinomial(2 * Q1, L - Qstar) + custom_logbinomial(L - Qstar, m - Qstar) + 0.50 * custom_logbinomial(2 * L, L + Qstar) - custom_logbinomial(2 * Q1 + L + Qstar + 1, L - Qstar) - 0.50 * custom_logbinomial(2 * L, L + m))

end

function generate_fourier_matrices(Qstar, N, L, Lz_list)

    @assert denominator(2 * L) == 1 && L ≥ abs(Qstar) "Invalid angular momentum."
    wigner_d_fourier_coeffecients = calculate_j_y_eigenstates(L)

    Q1 = (N - 1) // 2

    fourier_matrix = zeros(ComplexF64, length(Lz_list), numerator(1 + L - Qstar), numerator(2 * L + 1))

    for Lzprime in Qstar:1:L

        coeff = projection_coeff(L, Qstar, Q1, Lzprime) * (-1)^(round(Int64, Lzprime - Qstar))

        for (iter, Lz) in enumerate(Lz_list)

            for μ in -L:1:L

                fourier_matrix[iter, round(Int64, Lzprime + 1 - Qstar), round(Int64, μ + L + 1)] = wigner_d_fourier_coeffecients[(μ, Lzprime, Lz)]

            end

        end

        fourier_matrix[:, numerator(Lzprime + 1 - Qstar), :] .*= coeff

    end

    return fourier_matrix .* sqrt((2 * L + 1) / (4.0 * π))

end
end