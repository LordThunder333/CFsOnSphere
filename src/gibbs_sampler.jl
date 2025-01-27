include("projected_wavefunction.jl")
using .SpinPolarizedProjectedWavefunction
include("thermalization.jl")
using .Thermalization
include("monte_carlo_utilities.jl")
using .MonteCarloOnSphere

mutable struct MonteCarloState

    Ψ::Ψproj
    θ::Vector{Float64}
    ϕ::Vector{Float64}
    logpdf::Float64
    ### User needs to define here.
    Q::Rational{Int64}
    coulomb_energy::Float64
    accumulated_coulomb_energy::Float64
end

function Base.copy!(obj_current::MonteCarloState, obj_next::MonteCarloState, sampling_iter::Int64)
    copy!(obj_current.ψ, obj_next.ψ, sampling_iter)
    obj_current.θ[sampling_iter] = obj_next.θ[sampling_iter]
    obj_current.ϕ[sampling_iter] = obj_next.ϕ[sampling_iter]
    obj_current.logpdf = obj_next.logpdf
    obj_current.coulomb_energy = obj_next.coulomb_energy
    return
end

function Base.copy!(obj_current::ThermalizationState, obj_next::ThermalizationState)
    copy!(obj_current.ψ, obj_next.ψ)
    obj_current.θ .= obj_next.θ
    obj_current.ϕ .= obj_next.ϕ
    obj_current.logpdf = obj_next.logpdf
    return
end

function Base.copy(obj_current::MonteCarloState)
    return ThermalizationState(copy(obj_current.ψ), copy(obj_current.θ), copy(obj_current.ϕ), obj_current.logpdf)
end
