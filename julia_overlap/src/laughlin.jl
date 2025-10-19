"""
Laughlin wavefunction implementation in Julia
"""

using LinearAlgebra
using SpecialFunctions
include("math_utils.jl")

"""
Laughlin wavefunction for fractional quantum Hall states
Following the Python implementation more closely
"""
function laughlin_forward(electrons::Array{Float32, 2}, system_config)
    
    flux = system_config.flux
    nspins = system_config.nspins
    lz_center = system_config.lz_center
    
    # Use composite fermion approach like Python code
    cf_flux = 1  # Default value from Python
    n_electrons = sum(nspins)
    Q1 = flux / 2 - cf_flux * (n_electrons - 1)
    
    println("Debug Laughlin: flux=$flux, nspins=$nspins, n_electrons=$n_electrons")
    println("Debug Laughlin: Q1=$Q1, lz_center=$lz_center")
    
    # Convert to complex coordinates
    θ, φ = electrons[:, 1], electrons[:, 2]
    u = cos.(θ/2) .* exp.(0.5im * φ)  # Shape: (n_electrons,)
    v = sin.(θ/2) .* exp.(-0.5im * φ)  # Shape: (n_electrons,)
    
    println("Debug Laughlin: u range |u|=$(minimum(abs.(u))) to $(maximum(abs.(u)))")
    println("Debug Laughlin: v range |v|=$(minimum(abs.(v))) to $(maximum(abs.(v)))")
    
    # For now, let's implement a generic orbital construction 
    # that works for any filling by using the full orbital approach
    orbitals = laughlin_generic_orbitals(u, v, Q1, n_electrons)
    
    println("Debug Laughlin: orbitals shape=$(size(orbitals))")
    println("Debug Laughlin: orbitals finite? $(all(isfinite.(orbitals)))")
    
    # Compute determinant
    signs, logdets = slogdet(orbitals)
    
    println("Debug Laughlin: slogdet signs=$signs, logdets=$logdets")
    
    # Apply log-sum-exp trick (though here we only have one determinant)
    logmax = logdets  # Single determinant case
    log_wf = log(signs) + logmax
    
    println("Debug Laughlin: final log_wf=$log_wf")
    
    return log_wf
end

"""
Full (ground state) orbitals for Laughlin wavefunction
"""
function full_orbitals(u::Vector{ComplexF64}, v::Vector{ComplexF64}, Q::Float64)
    n = length(u)
    m_values = collect(-Q:1.0:Q)  # handles half-integers too
    # element = u v' - v u' + I
    element = (u * v') .- (v * u') .+ I*one(eltype(u)) |> z -> Matrix{ComplexF64}(z)
    # row-wise product -> (n, 1)
    jastrow = prod(element, dims=2)
    # orbitals: (n, #orbitals), broadcast jastrow over columns
    n_orb = length(m_values)
    orb = zeros(ComplexF64, n, n_orb)
    for (j, m) in enumerate(m_values)
        orb[:, j] .= (u .^ (Q + m)) .* (v .^ (Q - m)) .* vec(jastrow)
    end
    return orb
end

"""
Quasihole orbitals for Laughlin wavefunction
"""
function quasihole_orbitals(u::Vector{ComplexF64}, v::Vector{ComplexF64}, 
                           Q::Float64, excitation_lz::Float64)
    n_electrons = length(u)
    
    # Modified m values for quasihole state
    m_values_1 = collect(-Q:(excitation_lz-1))
    m_values_2 = collect(Q:-1:(excitation_lz+1))
    m_values = vcat(m_values_1, m_values_2)
    
    n_orbitals = length(m_values)
    
    # Compute Jastrow factor
    jastrow_elements = zeros(ComplexF64, n_electrons, n_electrons)
    for i in 1:n_electrons, j in 1:n_electrons
        if i == j
            jastrow_elements[i, j] = 1.0
        else
            jastrow_elements[i, j] = u[i] * v[j] - u[j] * v[i]
        end
    end
    
    jastrow = prod(jastrow_elements[i, j] for i in 1:n_electrons for j in (i+1):n_electrons)
    
    # Compute orbitals  
    orbitals = zeros(ComplexF64, n_electrons, n_orbitals)
    
    for (col, m) in enumerate(m_values)
        for i in 1:n_electrons
            orbitals[i, col] = (u[i]^(Q + m)) * (v[i]^(Q - m)) * jastrow
        end
    end
    
    return orbitals
end

"""
Quasiparticle orbitals for Laughlin wavefunction
"""
function quasiparticle_orbitals(u::Vector{ComplexF64}, v::Vector{ComplexF64}, 
                               Q::Float64, excitation_lz::Float64)
    n_electrons = length(u)
    m_values = collect(-Q:Q)
    n_orbitals = length(m_values) + 1  # Extra orbital for quasiparticle
    
    # Compute Jastrow factor and its derivatives
    jastrow_elements = zeros(ComplexF64, n_electrons, n_electrons)
    for i in 1:n_electrons, j in 1:n_electrons
        if i == j
            jastrow_elements[i, j] = 1.0
        else
            jastrow_elements[i, j] = u[i] * v[j] - u[j] * v[i]
        end
    end
    
    jastrow = prod(jastrow_elements[i, j] for i in 1:n_electrons for j in (i+1):n_electrons)
    
    # Compute derivatives for LLL projection
    jastrow_dv = zeros(ComplexF64, n_electrons)
    jastrow_du = zeros(ComplexF64, n_electrons)
    
    for i in 1:n_electrons
        # ∂jastrow/∂vᵢ ≈ jastrow * (∑ⱼ≠ᵢ -uⱼ/element[i,j] + uᵢ)
        sum_term_v = sum(j != i ? -u[j] / jastrow_elements[i, j] : 0.0 for j in 1:n_electrons)
        jastrow_dv[i] = jastrow * (sum_term_v + u[i])
        
        # ∂jastrow/∂uᵢ ≈ jastrow * (∑ⱼ≠ᵢ vⱼ/element[i,j] - vᵢ)
        sum_term_u = sum(j != i ? v[j] / jastrow_elements[i, j] : 0.0 for j in 1:n_electrons)
        jastrow_du[i] = jastrow * (sum_term_u - v[i])
    end
    
    # Standard orbitals
    orbitals = zeros(ComplexF64, n_electrons, n_orbitals)
    
    for (col, m) in enumerate(m_values)
        for i in 1:n_electrons
            orbitals[i, col] = (u[i]^(Q + m)) * (v[i]^(Q - m)) * jastrow
        end
    end
    
    # Excited orbital for quasiparticle
    m1 = excitation_lz
    for i in 1:n_electrons
        base_orbital = (u[i]^(Q + m1)) * (v[i]^(Q - m1))
        derivative_term = ((Q + 1 + m1) * v[i] * jastrow_dv[i] - 
                          (Q + 1 - m1) * u[i] * jastrow_du[i])
        orbitals[i, end] = base_orbital * derivative_term
    end
    
    return orbitals
end

"""
Generic orbital construction that handles various fillings
"""
function laughlin_generic_orbitals(u::Vector{ComplexF64}, v::Vector{ComplexF64}, Q1::Float64, n_electrons::Int)
    # Create a square matrix by choosing appropriate orbitals
    # For any n_electrons, we need exactly n_electrons orbitals
    
    # Use a simple approach: take the lowest |m| values centered around 0
    max_m = ceil(Int, (n_electrons - 1) / 2)
    m_values = collect(-max_m:max_m)[1:n_electrons]  # Take first n_electrons values
    
    println("Debug generic_orbitals: Q1=$Q1, n_electrons=$n_electrons")
    println("Debug generic_orbitals: m_values=$m_values")
    
    # Compute Jastrow factor (pairwise differences)
    jastrow_elements = zeros(ComplexF64, n_electrons, n_electrons)
    for i in 1:n_electrons, j in 1:n_electrons
        if i == j
            jastrow_elements[i, j] = 1.0
        else
            jastrow_elements[i, j] = u[i] * v[j] - u[j] * v[i]
        end
    end
    
    jastrow = prod(jastrow_elements[i, j] for i in 1:n_electrons for j in (i+1):n_electrons)
    
    println("Debug generic_orbitals: jastrow=$jastrow")
    println("Debug generic_orbitals: |jastrow|=$(abs(jastrow))")
    
    # Compute orbitals
    orbitals = zeros(ComplexF64, n_electrons, n_electrons)
    
    for (col, m) in enumerate(m_values)
        if col == 1  # Only debug the first column to reduce output
            println("Debug first column: m=$m, Q1=$Q1")
        end
        
        for i in 1:n_electrons
            # Use a fixed power based on Q1 and m
            power_u = Q1 + m
            power_v = Q1 - m
            
            if col == 1 && i <= 3  # Only debug first few elements of first column
                println("Debug: power_u=$power_u, power_v=$power_v for i=$i, m=$m")
                println("Debug: u[$i]=$(u[i]), |u[$i]|=$(abs(u[i]))")
                println("Debug: v[$i]=$(v[i]), |v[$i]|=$(abs(v[i]))")
            end
            
            # Safe complex exponentiation using exp(power * log(z))
            if abs(u[i]) > 1e-15
                log_u = log(u[i])
                term_u = exp(power_u * log_u)
                if col == 1 && i <= 3
                    println("Debug: log(u[$i])=$log_u")
                    println("Debug: power_u * log_u=$(power_u * log_u)")  
                    println("Debug: exp result=$term_u")
                end
            else
                term_u = ComplexF64(0.0)  # Handle u[i] ≈ 0 case
            end
            
            if abs(v[i]) > 1e-15
                log_v = log(v[i])
                term_v = exp(power_v * log_v)
                if col == 1 && i <= 3
                    println("Debug: log(v[$i])=$log_v")
                    println("Debug: power_v * log_v=$(power_v * log_v)")
                    println("Debug: exp result=$term_v")
                end
            else
                term_v = ComplexF64(0.0)  # Handle v[i] ≈ 0 case  
            end
            
            orbital_value = term_u * term_v * jastrow
            orbitals[i, col] = orbital_value
            
            if col == 1 && i <= 3
                println("Debug: term_u * term_v=$(term_u * term_v)")
                println("Debug: final orbitals[$i,$col]=$orbital_value")
                println("Debug: isfinite? $(isfinite(orbital_value))")
                println("---")
            end
        end
    end
    
    return orbitals
end