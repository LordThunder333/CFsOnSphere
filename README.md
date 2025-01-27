# CFsOnSphere

## Jain-Kamilla Projection in Spherical Geometry

This repository contains code implementing the **Jain-Kamilla (JK) projection** in spherical geometry, following the approach outlined in [this preprint](https://arxiv.org/abs/2412.09670). The primary application is the **lowest Landau level (LLL) projection** of composite fermion (CF) and, more generally, parton wavefunctions.

---

## Installation

To use this package, follow these steps:

1. **Install Julia**: Download and install Julia from [here](https://julialang.org/downloads/).

2. **Clone this Repository**:  
   This package is not yet registered with the Julia General Registry. To use it:
   - Clone the repository:
     ```bash
     git clone https://github.com/LordThunder333/CFsOnSphere.git
     ```
   - Navigate to the cloned folder.

3. **Activate the Environment**:
   - **If using Julia REPL**:  
     Activate the folder by running:
     ```julia
     ] activate .
     ```
     (This assumes you are in the cloned repository directory when opening the Julia REPL.)
   - **If running a Julia script**:  
     Run your script while activating the environment:
     ```bash
     julia --project=path_to_folder myscript.jl
     ```

---

## Usage

This code is designed to work with **single-component composite fermion wavefunctions**, with plans to extend its scope in the future. Below are the typical steps for using the code:

### 1. Identify Occupied Λ-level Orbitals
Determine all the Λ-level orbitals (Landau levels of composite fermions) needed for your calculation.

### 2. Construct the Wavefunction Object
Create a wavefunction object `ψ` using the constructor function:
```julia
Ψproj(Qstar, p, N, l_m_list)
```
- **Parameters**:
  - `Qstar`: Effective monopole strength (as a rational number).
  - `p`: Half the number of vortices bound to each electron.
  - `N`: Number of electrons in the system.
  - `l_m_list`: List of occupied Λ levels, represented as tuples `(L, Lz)` where `L` and `Lz` are rational numbers.

### 3. Update the Wavefunction
Before accessing the composite fermion orbitals for a given position, update the wavefunction:
- **For all particles**:
  ```julia
  update_wavefunction!(ψ, θ, φ)
  ```
  Updates the wavefunction assuming every particle is moved to the positions `(θ, φ)`.

- **For a single particle**:
  ```julia
  update_wavefunction!(ψ, θ[i], φ[i], i)
  ```
  Updates the wavefunction assuming only the `i`-th particle is moved to `(θ[i], φ[i])`.

### 4. Access Results
After updating the wavefunction:
- Log of the Jastrow factor:
  ```julia
  ψ.jastrow_factor_log
  ```
- Elements of the CF Slater determinant:
  ```julia
  ψ.slater_det
  ```
  Here, rows correspond to different orbitals, and columns correspond to different particles.

### 5. Explore Further
From this point, users can extend the functionality as needed for their specific calculations.

---

## Example Files
We strongly recommend first-time users, especially those new to **Monte Carlo methods** in Haldane’s spherical geometry or the **fractional quantum Hall effect**, to review the provided example files for guidance.
