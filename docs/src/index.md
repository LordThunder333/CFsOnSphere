# CFsOnSphere.jl Documentation

```@docs
    Ψproj(Qstar::Rational{Int64}, p::Int64, system_size::Int64, l_m_list::Vector{NTuple{2,Rational{Int64}}})
```
```@docs
    update_wavefunction!(Ψ::Ψproj, θ::Vector{Float64}, ϕ::Vector{Float64})
```
```@docs
    update_wavefunction!(Ψ::Ψproj, θ::Float64, ϕ::Float64, iter::Int64)
```
```@docs
    gibbs_thermalization!(RNG::AbstractRNG, Ψcurrent::Ψproj, Ψnext::Ψproj, θcurrent::Vector{Float64}, ϕcurrent::Vector{Float64}, θnext::Vector{Float64}, ϕnext::Vector{Float64}, σinit::Float64, logpdf::Function, num_thermalization::Int64) -> (Int64, Float64, Float64, Float64)
```
```@docs
    rand_θ_ϕ_gen(RNG, n_samples::Int)
```
```@docs
    proposal(RNG, θcurrent::Float64, ϕcurrent::Float64, σ::Float64)
```
```@docs
    construct_det_ratios(denominator_rows::Vector{Int64}, numerator_rows::Vector{Vector{Int64}})
```
```@docs
    legendre_polynomials!(res, x, kmax::Int64)
```
```@docs
    update_density!(θmesh::Vector{Float64}, θcurrent::Vector{Float64}, accumulated_density::Vector{Float64})
```
```@docs
    update_density!(θmesh::Vector{Float64}, ϕmesh::Vector{Float64}, θcurrent::Vector{Float64}, ϕcurrent::Vector{Float64}, accumulated_density::Matrix{Float64})
```