# Exact-parity Laughlin via Python Flax (keeps Julia laughlin.jl untouched)
using PyCall

if !isdefined(@__MODULE__, :_PY_LAUGH_CACHE)
    const _PY_LAUGH_CACHE = Dict{String, Any}()
end

"""
Exact-parity Laughlin forward using Python Flax via PyCall.
- electrons: (n_electrons, 2) Float32
- params_py: raw Python params object (same one passed to Psiformer)
- config_path: path to YAML, used to build Laughlin module (type= "laughlin")
"""
function laughlin_forward_py(electrons::Array{Float32, 2},
                             params_py::PyObject,
                             config_path::String)
    jax = pyimport("jax")
    np = pyimport("numpy")
    OmegaConf = pyimport("omegaconf").OmegaConf
    Config = pyimport("deephall.config").Config
    networks = pyimport("deephall.networks")
    dataclasses = pyimport("dataclasses")

    if !haskey(_PY_LAUGH_CACHE, config_path)
        cfg = Config.from_dict(OmegaConf.load(config_path))
        # replace network type with "laughlin" to match Python OverlapEstimator
        cfg_laugh = dataclasses.replace(cfg, network=dataclasses.replace(cfg.network, type="laughlin"))
        model = networks.make_network(cfg_laugh.system, cfg_laugh.network)
        netfunc = jax.jit(model.apply)
        _PY_LAUGH_CACHE[config_path] = netfunc
    end
    netfunc = _PY_LAUGH_CACHE[config_path]

    e_np = np.array(electrons, dtype=np.float32, copy=true)

    # Call exactly like Python: laughlin.apply(params, electrons)
    res = netfunc(params_py, e_np)

    # Safe extraction without triggering Julia array conversion
    builtins = pyimport("builtins")
    res_host = jax.device_get(res)
    res_py = PyObject(res_host)

    val_py = try
        pycall(res_py[:item], PyAny)  # 0-d numpy/jax arrays
    catch
        try
            pycall(builtins.complex, PyAny, res_py)  # coerce to Python complex scalar
        catch
            res_py  # fallback
        end
    end

    return ComplexF64(val_py)
end