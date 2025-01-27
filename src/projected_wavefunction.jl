module SpinPolarizedProjectedWavefunction
export Ψproj, update_wavefunction!
include("symmetric_polynomials.jl")
using .SymmetricPolynomials
include("jk_projection_utilities.jl")
using .JKProjection

function u_v_generator(θ, ϕ)
    
    return cos.(θ ./ 2) .* exp.(0.5im .* ϕ), sin.(θ ./ 2) .* exp.(-0.5im .* ϕ)

end

mutable struct Ψproj

    Qstar::Rational{Int64}
    p::Int64
    system_size::Int64

    l_m_list::Vector{NTuple{2,Rational{Int64}}} ### This is a list of tuples of the form (L, Lz)
    Lmax::Rational{Int64}
    μ_list::Vector{Rational{Int64}}
    Lz_list::Vector{Rational{Int64}}

    fourier_tot_matrix::Matrix{ComplexF64} ### Reshaped fourier matrix for efficient matrix multiplication.

    U::Vector{ComplexF64}
    V::Vector{ComplexF64}

    exp_θ::Matrix{ComplexF64}
    exp_ϕ::Matrix{ComplexF64}

    dist_matrix::Matrix{Float64}
    u_v_ratio_matrix::Matrix{ComplexF64}

    elementary_symmetric_polynomials::Matrix{ComplexF64}
    reg_coeffs::Vector{Float64}

    wigner_d_matrices::Matrix{ComplexF64}
    wigner_D_matrices::Array{ComplexF64,3}

    jastrow_factor_log::ComplexF64
    slater_det::Matrix{ComplexF64}
end

function Ψproj(Qstar::Rational{Int64}, p::Int64, system_size::Int64, l_m_list::Vector{NTuple{2,Rational{Int64}}})
    
    Lmax = maximum(first, l_m_list)

    fourier_matrix = zeros(ComplexF64, length(l_m_list), numerator(1 + Lmax-Qstar), numerator(1+2*Lmax))

    Lgrid = unique(first.(l_m_list))
    liters = [findall(x->x[1]==L, l_m_list) for L in Lgrid]
    
    for (Liter, L) in enumerate(Lgrid)

        fourier_matrix[liters[Liter], begin:begin+numerator(L-Qstar), numerator(1-L+Lmax):1:numerator(1+L+Lmax)] .= generate_fourier_matrices(Qstar, system_size, L, last.(l_m_list[liters[Liter]]))

    end

    fourier_tot_matrix = reshape(fourier_matrix, :, numerator(1+2*Lmax))

    Lz_list = last.(l_m_list)
    
    μ_list = collect(-Lmax:1:Lmax)

    U = zeros(ComplexF64, system_size)
    V = zeros(ComplexF64, system_size)

    exp_θ = zeros(ComplexF64, length(μ_list), system_size)
    exp_ϕ = zeros(ComplexF64, length(l_m_list), system_size)

    jastrow_factor_log = 0.0 + 0.0im

    slater_det = zeros(ComplexF64, length(l_m_list), system_size)
    dist_matrix = zeros(Float64, system_size - 1, system_size)

    u_v_ratio_matrix = zeros(ComplexF64, system_size - 1, system_size)
    elementary_symmetric_polynomials = zeros(ComplexF64, 1 + round(Int64, Lmax - Qstar), system_size)

    wigner_d_matrices = zeros(ComplexF64, length(l_m_list) * (1 + numerator(Lmax - Qstar)), system_size)
    wigner_D_matrices = zeros(ComplexF64, length(l_m_list), 1 + numerator(Lmax - Qstar), system_size)

    reg_coeffs = zeros(Float64, round(Int64, Lmax - Qstar))
    
    for i in eachindex(reg_coeffs)
        reg_coeffs[i] = (i/(system_size-i))
    end

    return Ψproj(Qstar, p, system_size, l_m_list, Lmax, μ_list, Lz_list, fourier_tot_matrix, U, V, exp_θ, exp_ϕ, dist_matrix, u_v_ratio_matrix, elementary_symmetric_polynomials, reg_coeffs, wigner_d_matrices, wigner_D_matrices, jastrow_factor_log, slater_det)
end

function update_wavefunction!(Ψ::Ψproj, θ::Vector{Float64}, ϕ::Vector{Float64})

    Ψ.exp_θ .= exp.(-1.0im .* Ψ.μ_list * transpose(θ))
    Ψ.exp_ϕ .= exp.(1.0im .* Ψ.Lz_list * transpose(ϕ))

    Ψ.jastrow_factor_log = zero(ComplexF64)

    Ψ.U, Ψ.V = u_v_generator(θ, ϕ)

    δu = zero(ComplexF64)
    δv = zero(ComplexF64)

    for i = 1:Ψ.system_size-1
        for j = i+1:Ψ.system_size

            δu = conj(Ψ.U[i]) * Ψ.U[j] + conj(Ψ.V[i]) * Ψ.V[j]
            δv = Ψ.U[i] * Ψ.V[j] - Ψ.V[i] * Ψ.U[j]

            Ψ.u_v_ratio_matrix[j-1, i] = δu / δv
            Ψ.u_v_ratio_matrix[i, j] = -conj(δu) / δv

            Ψ.jastrow_factor_log += 2.0 * Ψ.p * log(δv)

            Ψ.dist_matrix[j-1, i] = 2.0 * abs(δv)
            Ψ.dist_matrix[i, j] = Ψ.dist_matrix[j-1, i]

        end
    end

    @simd for electron_iter in axes(Ψ.elementary_symmetric_polynomials, 2)

        @inbounds @views get_symmetric_polynomials!(Ψ.elementary_symmetric_polynomials[:, electron_iter], Ψ.u_v_ratio_matrix[:, electron_iter], numerator(Ψ.Lmax-Ψ.Qstar), Ψ.reg_coeffs)

    end

    Ψ.wigner_d_matrices .= Ψ.fourier_tot_matrix * Ψ.exp_θ
    Ψ.wigner_D_matrices .= reshape(Ψ.wigner_d_matrices, size(Ψ.wigner_D_matrices))
    
    @simd for iter in axes(Ψ.wigner_D_matrices, 3)

        @inbounds @views Ψ.wigner_D_matrices[:, :, iter] .*= Ψ.exp_ϕ[:, iter]

    end

    @simd for iter in axes(Ψ.slater_det, 2)

        @inbounds @views Ψ.slater_det[:, iter] .= Ψ.wigner_D_matrices[:, :, iter] * Ψ.elementary_symmetric_polynomials[:, iter]

    end

    return
end

function update_wavefunction!(Ψ::Ψproj, θ::Float64, ϕ::Float64, iter::Int64)

    Ψ.exp_θ[:, iter] .= exp.(-1.0im .* Ψ.μ_list * θ)
    Ψ.exp_ϕ[:, iter] .= exp.(1.0im .* Ψ.Lz_list * ϕ)

    unew, vnew = u_v_generator(θ, ϕ)

    δv_old = zero(ComplexF64)
    δv_new = zero(ComplexF64)

    δu_new = zero(ComplexF64)

    for i = 1:Ψ.system_size

        if i < iter

            δv_old = Ψ.U[i] * Ψ.V[iter] - Ψ.V[i] * Ψ.U[iter]
            δv_new = Ψ.U[i] * vnew -  Ψ.V[i] * unew

            δu_new = conj(Ψ.U[i]) * unew + conj(Ψ.V[i]) * vnew

            Ψ.u_v_ratio_matrix[iter-1, i] = δu_new / δv_new
            Ψ.u_v_ratio_matrix[i, iter] = -conj(δu_new) / δv_new

            Ψ.jastrow_factor_log += 2.0 * Ψ.p * log(δv_new / δv_old)

            Ψ.dist_matrix[iter-1, i] = 2.0 * abs(δv_new)
            Ψ.dist_matrix[i, iter] = Ψ.dist_matrix[iter-1, i]

        elseif i > iter

            δv_old = -Ψ.U[i] * Ψ.V[iter] + Ψ.V[i] * Ψ.U[iter]
            δv_new = -Ψ.U[i] * vnew +  Ψ.V[i] * unew

            δu_new = (Ψ.U[i]) * conj(unew) + (Ψ.V[i]) * conj(vnew)

            Ψ.u_v_ratio_matrix[i-1, iter] = δu_new / δv_new
            Ψ.u_v_ratio_matrix[iter, i] = -conj(δu_new) / δv_new

            Ψ.jastrow_factor_log += 2.0 * Ψ.p * log(δv_new / δv_old)

            Ψ.dist_matrix[i-1, iter] = 2.0 * abs(δv_new)
            Ψ.dist_matrix[iter, i] = Ψ.dist_matrix[i-1, iter]

        end

    end

    Ψ.U[iter], Ψ.V[iter] = unew, vnew

    @simd for electron_iter in axes(Ψ.elementary_symmetric_polynomials, 2)

        @inbounds @views get_symmetric_polynomials!(Ψ.elementary_symmetric_polynomials[:, electron_iter], Ψ.u_v_ratio_matrix[:, electron_iter], numerator(Ψ.Lmax-Ψ.Qstar), Ψ.reg_coeffs)

    end

    Ψ.wigner_d_matrices[:, iter] .= Ψ.fourier_tot_matrix * Ψ.exp_θ[:, iter]
    
    @simd for j in axes(Ψ.wigner_D_matrices, 2)
        @inbounds @views Ψ.wigner_D_matrices[:, j, iter] .= Ψ.wigner_d_matrices[1+size(Ψ.wigner_D_matrices, 1)*(j-1):size(Ψ.wigner_D_matrices, 1)*(j), iter]
    end

    Ψ.wigner_D_matrices[:, :, iter] .*= Ψ.exp_ϕ[:, iter]

    @simd for electron_iter in axes(Ψ.slater_det, 2)

        @inbounds @views Ψ.slater_det[:, electron_iter] .= Ψ.wigner_D_matrices[:, :, electron_iter] * Ψ.elementary_symmetric_polynomials[:, electron_iter]

    end

    return
end

function Base.copy!(Ψ1::Ψproj, Ψ2::Ψproj)

    Ψ1.dist_matrix .= Ψ2.dist_matrix

    Ψ1.exp_θ .= Ψ2.exp_θ
    Ψ1.exp_ϕ .= Ψ2.exp_ϕ
    Ψ1.U .= Ψ2.U
    Ψ1.V .= Ψ2.V

    Ψ1.jastrow_factor_log = Ψ2.jastrow_factor_log
    Ψ1.slater_det .= Ψ2.slater_det

    Ψ1.elementary_symmetric_polynomials .= Ψ2.elementary_symmetric_polynomials
    Ψ1.u_v_ratio_matrix .= Ψ2.u_v_ratio_matrix

    Ψ1.wigner_d_matrices .= Ψ2.wigner_d_matrices
    Ψ1.wigner_D_matrices .= Ψ2.wigner_D_matrices

    return
end

function Base.copy(Ψ1::Ψproj)
    return Ψproj(Ψ1.Qstar, Ψ1.p, Ψ1.system_size, copy(Ψ1.l_m_list), Ψ1.Lmax, copy(Ψ1.μ_list), copy(Ψ1.Lz_list), copy(Ψ1.fourier_tot_matrix), copy(Ψ1.U), copy(Ψ1.V), copy(Ψ1.exp_θ), copy(Ψ1.exp_ϕ), copy(Ψ1.dist_matrix), copy(Ψ1.u_v_ratio_matrix), copy(Ψ1.elementary_symmetric_polynomials), copy(Ψ1.reg_coeffs), copy(Ψ1.wigner_d_matrices), copy(Ψ1.wigner_D_matrices), Ψ1.jastrow_factor_log, copy(Ψ1.slater_det))
end

function Base.copy!(Ψ1::Ψproj, Ψ2::Ψproj, iter::Int64)

    Ψ1.dist_matrix .= Ψ2.dist_matrix

    Ψ1.exp_θ[:, iter] .= Ψ2.exp_θ[:, iter]
    Ψ1.exp_ϕ[:, iter] .= Ψ2.exp_ϕ[:, iter]
    Ψ1.U[iter] = Ψ2.U[iter]
    Ψ1.V[iter] = Ψ2.V[iter]

    Ψ1.jastrow_factor_log = Ψ2.jastrow_factor_log
    Ψ1.slater_det .= Ψ2.slater_det

    Ψ1.elementary_symmetric_polynomials .= Ψ2.elementary_symmetric_polynomials

    Ψ1.u_v_ratio_matrix .= Ψ2.u_v_ratio_matrix

    Ψ1.wigner_d_matrices[:, iter] .= Ψ2.wigner_d_matrices[:, iter]
    Ψ1.wigner_D_matrices[:, :, iter] .= Ψ2.wigner_D_matrices[:, :, iter]

    return
end
end