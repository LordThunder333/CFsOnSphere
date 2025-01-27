module MonteCarloOnSphere
using CoordinateTransformations
using Quaternionic
using StaticArrays
using LinearAlgebra
export rand_θ_ϕ_gen, proposal, arm_parameters, arm_scale_factor

"""
    rand_θ_ϕ_gen(RNG, n_samples::Int) -> Tuple{Vector{Float64}, Vector{Float64}}

Generate random spherical coordinates (θ,ϕ) uniformly distributed on a unit sphere.

# Arguments
- `RNG`: Random number generator
- `n_samples::Int`: Number of random samples to generate

# Returns
- `θlist::Vector{Float64}`: Array of θ values in [0,π]
- `ϕlist::Vector{Float64}`: Array of ϕ values in (-π,π]

"""
function rand_θ_ϕ_gen(RNG, n_samples::Int)
    Xmat = randn(RNG, Float64, 3, n_samples)
    θlist = zeros(Float64, n_samples)
    ϕlist = zeros(Float64, n_samples)
    @simd for i in axes(Xmat, 2)
        # x = randn(RNG, Float64, 3)
        @inbounds @views sph  = SphericalFromCartesian()(Xmat[:, i])
        θlist[i], ϕlist[i] = pi/2-sph.ϕ, sph.θ
    end
    return θlist, ϕlist
end


"""
    proposal(RNG, θcurrent::Float64, ϕcurrent::Float64, σ::Float64) -> (θnew::Float64, ϕnew::Float64)

Generate a proposed new position on a sphere for a Monte Carlo step, given the current position (θcurrent, ϕcurrent).

The function generates a new position on the sphere using the following steps:
1. Creates a random displacement using a Gaussian step size (σ) and random direction
2. Represents this displacement as a quaternion from the north pole
3. Uses quaternion rotation to map the current position to the proposal position
4. Converts the result back to spherical coordinates

This method ensures uniform sampling across the sphere.

# Arguments
- `RNG`: Random number generator
- `θcurrent`: Current polar angle θ ∈ [0, π]
- `ϕcurrent`: Current azimuthal angle ϕ ∈ [-π, π]
- `σ`: Standard deviation of the Gaussian distribution for step size

# Returns
A tuple containing the new proposed position (θnew, ϕnew) on the sphere.

Note: The angles follow the mathematical physics convention where θ is the polar angle 
from the z-axis and ϕ is the azimuthal angle in the x-y plane.
"""
function proposal(RNG, θcurrent::Float64, ϕcurrent::Float64, σ::Float64)

    δθ = randn(RNG) * σ
    δϕ = rand(RNG) * (2.0 * pi) - pi

    sδθ, cδθ = sincos(δθ)
    sδϕ, cδϕ = sincos(δϕ)

    v = Quaternion(sδθ * cδϕ, sδθ * sδϕ, cδθ)

    sϕ, cϕ = sincos(ϕcurrent)

    q = exp(Quaternion(-sϕ, cϕ, 0.0) * θcurrent / 2)
    v = q * v * inv(q)
    x = SA[v[2], v[3], v[4]]

    sph = SphericalFromCartesian()(x)
    return pi/2-sph.ϕ, sph.θ
end

"""
Returns parameters for ARM scheme for step size adapation to maintain acceptance ratio.
"""
function arm_parameters(ideal_acceptance_ratio::Float64, r::Float64)
    a = 1.0
    b = 0.0
    for i = 1:1000
        c = (a * ideal_acceptance_ratio + b)^r
        a = (a * ideal_acceptance_ratio + b)^(1 / r) - c
        b = c
    end
    return a, b
end
"""
Returns ARM scale factor for a given acceptance ratio, the ideal acceptance ratio and ARM parameters.
"""
function arm_scale_factor(p, p_i, a, b)
    return log(a * p_i + b) / log(a * p + b)
end
end
