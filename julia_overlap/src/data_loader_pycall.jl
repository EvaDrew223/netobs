"""
Data loading using PyCall to handle Python objects in NPZ files
"""

using YAML
using PyCall
using LinearAlgebra

# Import Python modules when needed
function get_numpy()
    try
        return pyimport("numpy")
    catch e
        error("Failed to import numpy: $e")
    end
end

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
Checkpoint state using PyCall
"""
struct CheckpointState
    step::Int
    params_py::PyObject
    data::Array{Float32, 3}
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
Load checkpoint using PyCall
"""
function load_checkpoint_pycall(checkpoint_path::String)
    np = get_numpy()

    py_data = np.load(checkpoint_path, allow_pickle=true)

    step_arr = PyCall.pycall(py_data.__getitem__, PyCall.PyObject, "step")
    step = Int(step_arr.item())

    data_arr = PyCall.pycall(py_data.__getitem__, PyCall.PyObject, "data")
    data = convert(Array{Float32, 3}, data_arr)

    mcmc_width_arr = PyCall.pycall(py_data.__getitem__, PyCall.PyObject, "mcmc_width")
    mcmc_width = Float32(mcmc_width_arr.item())

    # Keep params as raw Python object for exact Flax parity
    params_arr = PyCall.pycall(py_data.__getitem__, PyCall.PyObject, "params")
    params_py = params_arr.item()

    py_data.close()

    return CheckpointState(step, params_py, data, mcmc_width)
end

"""
Convert Python parameter objects to Julia arrays
"""
function convert_python_params(py_params)
    function convert_recursive(obj)
        try
            # Check if it's a dictionary-like object
            if PyCall.hasproperty(obj, "keys") && PyCall.hasproperty(obj, "__getitem__")
                result = Dict{String, Any}()
                for key in obj.keys()
                    key_str = string(key)
                    result[key_str] = convert_recursive(obj[key])
                end
                return result
            # Check if it has array interface
            elseif PyCall.hasproperty(obj, "__array__")
                # JAX arrays have __array__ method
                np_array = obj.__array__()
                return Array(np_array)
            # Try direct conversion to array
            else
                return Array(obj)
            end
        catch e
            # If all else fails, return as-is and let caller handle
            println("Warning: Could not convert parameter: $(typeof(obj))")
            return obj
        end
    end
    
    return convert_recursive(py_params)
end