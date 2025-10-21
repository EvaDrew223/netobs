## Psiformer network implementation in Julia (PyCall-backed forwards)

using LinearAlgebra
using Statistics
using PyCall

if !isdefined(@__MODULE__, :_PY_NET_CACHE)
    const _PY_NET_CACHE = Dict{String, Any}()
end

include("math_utils.jl")

# Single-sample forward via Python Flax (already existed)
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

    jax = pyimport("jax")
    builtins = pyimport("builtins")
    res_host = jax.device_get(res)
    res_py = PyObject(res_host)
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