"""
Data loading functions for checkpoint and configuration files
"""

using YAML
using NPZ
using LinearAlgebra

"""
Structure to hold system configuration
"""
struct SystemConfig
    flux::Float64
    radius::Union{Float64, Nothing}
    nspins::Vector{Int}
    interaction_strength::Float64
    lz_center::Float64
    interaction_type::String
end

"""
Structure to hold network configuration
"""
struct NetworkConfig
    type::String
    orbital::String
    psiformer_num_heads::Int
    psiformer_heads_dim::Int
    psiformer_num_layers::Int
    psiformer_determinants::Int
end

"""
Structure to hold checkpoint state
"""
struct CheckpointState
    step::Int
    params::Dict{String, Any}
    data::Array{Float32, 3}  # (n_samples, n_electrons, ndim)
    mcmc_width::Float32
end

"""
Load configuration from YAML file
"""
function load_config(config_path::String)
    config_dict = YAML.load_file(config_path)
    
    system = config_dict["system"]
    network = config_dict["network"]
    
    sys_config = SystemConfig(
        system["flux"],
        system["radius"],
        system["nspins"],
        system["interaction_strength"],
        system["lz_center"],
        system["interaction_type"]
    )
    
    net_config = NetworkConfig(
        network["type"],
        network["orbital"],
        network["psiformer"]["num_heads"],
        network["psiformer"]["heads_dim"],
        network["psiformer"]["num_layers"],
        network["psiformer"]["determinants"]
    )
    
    return sys_config, net_config
end

"""
Load checkpoint from NPZ file
"""
function load_checkpoint(checkpoint_path::String)
    data = npzread(checkpoint_path)
    
    step = Int(data["step"])
    params = data["params"]
    electron_data = data["data"]
    mcmc_width = Float32(data["mcmc_width"])
    
    return CheckpointState(step, params, electron_data, mcmc_width)
end

"""
Convert nested Python dictionary parameters to Julia format
"""
function convert_params(py_params::Dict{String, Any})
    julia_params = Dict{String, Any}()
    
    for (key, value) in py_params
        if isa(value, Dict)
            julia_params[key] = convert_params(value)
        else
            julia_params[key] = value
        end
    end
    
    return julia_params
end