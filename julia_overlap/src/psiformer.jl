"""
Psiformer network implementation in Julia
"""

using LinearAlgebra
using Statistics
using PyCall

const _PY_NET_CACHE = Dict{String, Any}()

"""
Convert nested Dict{Any, Any} to Dict{String, Any} recursively
"""
function convert_dict_types(d)
    if isa(d, Dict)
        return Dict{String, Any}(string(k) => convert_dict_types(v) for (k, v) in d)
    else
        return d
    end
end

"""
Convert PyObject array to Julia array
"""
function convert_pyobject(obj)
    if isa(obj, PyObject)
        try
            # Convert JAX array to host, then to numpy, then to Julia
            jax = pyimport("jax")
            np = pyimport("numpy")
            
            # Move from device to host
            host_array = jax.device_get(obj)
            
            # Convert to numpy array with appropriate dtype and make writable copy
            # Check if it's complex or real using Python's hasattr
            py_builtin = pyimport("builtins")
            if py_builtin.hasattr(host_array, "dtype") && occursin("complex", string(host_array.dtype))
                # Handle complex arrays - use complex64 for JAX compatibility
                np_array = np.array(host_array, dtype=np.complex64, copy=true)  # Force copy to make writable
                # Use PyCall's convert method
                return convert(Array{ComplexF32}, np_array)
            else
                # Handle real arrays
                np_array = np.array(host_array, dtype=np.float32, copy=true)  # Force copy to make writable
                # Use PyCall's convert method
                return convert(Array{Float32}, np_array)
            end
        catch e
            println("Warning: Could not convert PyObject: $e")
            # Fallback: force conversion to numpy array with copy and then to Julia
            try
                np = pyimport("numpy")
                jax = pyimport("jax") 
                # Force device_get first
                host_obj = jax.device_get(obj)
                # Force to numpy float32 with writable copy
                np_array = np.array(host_obj, dtype=np.float32, copy=true)
                return convert(Array{Float32}, np_array)
            catch e2
                println("Fallback conversion also failed: $e2")
                error("Failed to convert PyObject to Julia array: $obj")
            end
        end
    else
        return obj
    end
end
include("math_utils.jl")

"""
Convert electron positions from spherical to input features
"""
function input_features(electrons::Array{Float32, 2}, spins::Vector{Float64})
    n_electrons = size(electrons, 1)
    θ, φ = electrons[:, 1], electrons[:, 2]
    
    features = hcat(
        cos.(θ),
        sin.(θ) .* cos.(φ),
        sin.(θ) .* sin.(φ),
        spins
    )
    
    return features
end

"""
Jastrow factor computation
"""
function compute_jastrow(electrons::Array{Float32, 2}, nspins::Vector{Int}, 
                        ee_par::Float32, ee_anti::Float32)
    n_electrons = size(electrons, 1)
    θ, φ = electrons[:, 1], electrons[:, 2]
    
    # Convert to Cartesian coordinates
    cart_e = hcat(
        cos.(θ),
        sin.(θ) .* cos.(φ),
        sin.(θ) .* sin.(φ)
    )
    
    # Compute pairwise distances
    r_ee = zeros(n_electrons, n_electrons)
    for i in 1:n_electrons, j in 1:n_electrons
        if i != j
            diff = cart_e[i, :] - cart_e[j, :]
            r_ee[i, j] = norm(diff)
        end
    end
    
    # Parallel spin pairs (same spin)
    jastrow_par = 0.0
    for i in 1:(nspins[1]-1), j in (i+1):nspins[1]
        r = r_ee[i, j]
        jastrow_par += -(0.25 * ee_par^2) / (ee_par + r)
    end
    
    if nspins[2] > 1
        offset = nspins[1]
        for i in 1:(nspins[2]-1), j in (i+1):nspins[2]
            r = r_ee[offset + i, offset + j]
            jastrow_par += -(0.25 * ee_par^2) / (ee_par + r)
        end
    end
    
    # Antiparallel spin pairs (opposite spin)
    jastrow_anti = 0.0
    if nspins[1] > 0 && nspins[2] > 0
        for i in 1:nspins[1], j in 1:nspins[2]
            r = r_ee[i, nspins[1] + j]
            jastrow_anti += -(0.5 * ee_anti^2) / (ee_anti + r)
        end
    end
    
    return jastrow_par + jastrow_anti
end

"""
Compute monopole harmonics orbitals
"""
function compute_orbitals(h_one::Matrix{ComplexF64}, θ::Vector{Float32}, φ::Vector{Float32}, 
                         Q::Float64, nspins::Vector{Int}, ndets::Int,
                         orbital_weights::Array{ComplexF64, 4})
    
    n_electrons = length(θ)
    m_values = collect(-Q:Q)
    n_orbitals = length(m_values)
    
    # Compute normalization factors
    norm_factors = [sqrt(binomial_coeff(2*Q, Q - m)) for m in m_values]
    
    # Compute u and v from spherical coordinates
    u = cos.(θ/2) .* exp.(0.5im * φ)  # Shape: (n_electrons,)
    v = sin.(θ/2) .* exp.(-0.5im * φ)  # Shape: (n_electrons,)
    
    # Compute envelope: norm_factor * u^(Q+m) * v^(Q-m)
    envelope = zeros(ComplexF64, n_electrons, n_orbitals)
    for (i, m) in enumerate(m_values)
        envelope[:, i] = norm_factors[i] * (u .^ (Q + m)) .* (v .^ (Q - m))
    end
    
    # Apply the learned orbital weights
    # orbital_weights shape: (n_orbitals, num_heads, n_electrons, ndets)
    orbitals = zeros(ComplexF64, ndets, n_electrons, n_electrons)  
    
    for det in 1:ndets
        for i in 1:n_electrons
            for j in 1:n_electrons
                # Weight the envelope with learned parameters
                weighted_envelope = sum(orbital_weights[:, :, i, det] .* envelope[j, :])
                orbitals[det, i, j] = weighted_envelope
            end
        end
    end
    
    return orbitals
end

"""
Psiformer layers (transformer-like architecture)
"""
function psiformer_layers(electrons::Array{Float32, 2}, spins::Vector{Float64},
                         params::Dict{String, Any}, 
                         num_heads::Int, num_layers::Int, heads_dim::Int)
    
    n_electrons = size(electrons, 1)
    attention_dim = num_heads * heads_dim
    
    # Input features
    h_one = ComplexF64.(input_features(electrons, spins))
    
    # Initial projection
    dense0_w = ComplexF64.(convert_pyobject(params["Dense_0"]["kernel"]))
    h_one = h_one * dense0_w
    
    # Transformer layers
    for layer in 0:(num_layers-1)
        # Multi-head attention
        attn_params = params["MultiHeadAttention_$layer"]
        W_q = ComplexF64.(convert_pyobject(attn_params["query"]["kernel"]))
        b_q = ComplexF64.(convert_pyobject(attn_params["query"]["bias"]))
        W_k = ComplexF64.(convert_pyobject(attn_params["key"]["kernel"])) 
        b_k = ComplexF64.(convert_pyobject(attn_params["key"]["bias"]))
        W_v = ComplexF64.(convert_pyobject(attn_params["value"]["kernel"]))
        b_v = ComplexF64.(convert_pyobject(attn_params["value"]["bias"]))
        W_o = ComplexF64.(convert_pyobject(attn_params["out"]["kernel"]))
        b_o = ComplexF64.(convert_pyobject(attn_params["out"]["bias"]))
        
        attn_out = multi_head_attention(h_one, W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o,
                                       num_heads, heads_dim)
        
        # Add & norm
        dense1_w = ComplexF64.(convert_pyobject(params["Dense_$(2*layer + 1)"]["kernel"]))
        h_one = h_one + attn_out * dense1_w
        
        # Layer norm
        ln_scale = ComplexF64.(convert_pyobject(params["LayerNorm_$(2*layer)"]["scale"]))
        ln_bias = ComplexF64.(convert_pyobject(params["LayerNorm_$(2*layer)"]["bias"]))
        h_one = layer_norm(h_one) .* ln_scale' .+ ln_bias'
        
        # Feed forward
        dense2_w = ComplexF64.(convert_pyobject(params["Dense_$(2*layer + 2)"]["kernel"]))
        dense2_b = ComplexF64.(convert_pyobject(params["Dense_$(2*layer + 2)"]["bias"]))
        ff_out = apply_activation(h_one * dense2_w .+ dense2_b', "tanh")
        
        # Add & norm
        h_one = h_one + ff_out
        ln2_scale = ComplexF64.(convert_pyobject(params["LayerNorm_$(2*layer + 1)"]["scale"]))
        ln2_bias = ComplexF64.(convert_pyobject(params["LayerNorm_$(2*layer + 1)"]["bias"]))
        h_one = layer_norm(h_one) .* ln2_scale' .+ ln2_bias'
    end
    
    return h_one
end

"""
Exact-parity Psiformer forward using Python Flax via PyCall.
- electrons: (n_electrons, 2) Float32
- params_py: raw Python params object from NPZ
- config_path: path to the YAML config used to build the model
"""
function psiformer_forward(electrons::Array{Float32, 2}, params_py::PyObject, config_path::String)
    jax = pyimport("jax")
    np = pyimport("numpy")
    OmegaConf = pyimport("omegaconf").OmegaConf
    Config = pyimport("deephall.config").Config
    networks = pyimport("deephall.networks")

    if !haskey(_PY_NET_CACHE, config_path)
        cfg = Config.from_dict(OmegaConf.load(config_path))
        model = networks.make_network(cfg.system, cfg.network)
        netfunc = jax.jit(model.apply)
        _PY_NET_CACHE[config_path] = netfunc
    end
    netfunc = _PY_NET_CACHE[config_path]

    e_np = np.array(electrons, dtype=np.float32, copy=true)
    res = netfunc(params_py, e_np)

    # Bring to host but keep as PyObject to avoid premature Julia conversion
    jax = pyimport("jax")
    builtins = pyimport("builtins")

    res_host = jax.device_get(res)              # keep as Python object
    res_py = PyObject(res_host)

    # Try 0-d array/scalar .item(); fall back to Python complex() coercion; finally return as-is
    val_py = try
        pycall(res_py[:item], PyAny)
    catch
        try
            pycall(builtins.complex, PyAny, res_py)
        catch
            res_py
        end
    end

    return ComplexF64(val_py)
end