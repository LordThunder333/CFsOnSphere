module CFsOnSphere

include("projected_wavefunction.jl")
using .SpinPolarizedProjectedWavefunction

include("monte_carlo_utilities.jl")
using .MonteCarloOnSphere

include("legendre_polynomials.jl")
using .LegendrePolynomials

using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

using JLD2

export Ψproj, update_wavefunction!, rand_θ_ϕ_gen, proposal, arm_parameters, arm_scale_factor, legendre_polynomials!, save, load, logdet, lu, inv, update_density!

function update_density!(θmesh::Vector{Float64}, θcurrent::Vector{Float64}, accumulated_density::Vector{Float64})
    
    for θ in θcurrent
        accumulated_density[searchsortedfirst(θmesh, θ)-1] += 1.0
    end

    return

end

function update_density!(θmesh::Vector{Float64}, ϕmesh::Vector{Float64}, θcurrent::Vector{Float64}, ϕcurrent::Vector{Float64}, accumulated_density::Matrix{Float64})
    
    for iter in eachindex(θcurrent)
        accumulated_density[searchsortedfirst(θmesh, θcurrent[iter])-1, searchsortedfirst(ϕmesh, ϕcurrent[iter])-1] += 1.0
    end

    return

end


end
