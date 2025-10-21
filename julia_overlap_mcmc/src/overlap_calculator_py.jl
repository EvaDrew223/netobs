## Overlap calculation between Psiformer and Laughlin wavefunctions
## via pycall laughlin + psiformer backends
## without MCMC sampling parity
## Only does single-shot evaluation (used by --no-mcmc flag)

using LinearAlgebra
using Statistics
using Distributed

include("data_loader_pycall.jl")
include("psiformer.jl")      # psiformer_forward (PyCall-backed)
include("laughlin_py.jl")    # laughlin_forward_py (PyCall-backed)
include("math_utils.jl")

function compute_log_ratio_py(electrons::Array{Float32, 2},
                              params_py::PyObject,
                              system_config, network_config,
                              config_path::String)
    logpsi = psiformer_forward(electrons, params_py, config_path)
    logphi = laughlin_forward_py(electrons, params_py, config_path)
    return logphi - logpsi
end

function compute_batch_overlap_py(electrons_batch::Array{Float32, 3},
                                  params_py::PyObject,
                                  system_config, network_config,
                                  config_path::String;
                                  use_parallel::Bool=true)
    batch_size = size(electrons_batch, 1)
    log_ratios = zeros(ComplexF64, batch_size)

    if use_parallel && nprocs() > 1
        log_ratios = pmap(1:batch_size) do i
            electron_config = electrons_batch[i, :, :]
            compute_log_ratio_py(electron_config, params_py, system_config, network_config, config_path)
        end
    else
        for i in 1:batch_size
            electron_config = electrons_batch[i, :, :]
            log_ratios[i] = compute_log_ratio_py(electron_config, params_py, system_config, network_config, config_path)
        end
    end

    shift = mean(log_ratios)
    ratios = exp.(log_ratios .- shift)
    ratio_squares = abs.(ratios).^2
    return ratios, ratio_squares
end

function calculate_overlap_py(checkpoint_path::String, config_path::String;
                              steps::Int=50, use_parallel::Bool=true)
    println("Loading configuration from: $config_path")
    system_config, network_config = load_config(config_path)

    println("Loading checkpoint from: $checkpoint_path")
    checkpoint = load_checkpoint_pycall(checkpoint_path)

    all_electrons = checkpoint.data
    total_samples = size(all_electrons, 1)
    println("Total available samples: $total_samples")

    if steps > total_samples
        println("Warning: Requested steps ($steps) > available samples ($total_samples)")
        steps = total_samples
    end

    sample_indices = steps == 1 ? [1] : round.(Int, range(1, total_samples, length=steps))
    selected_electrons = all_electrons[sample_indices, :, :]

    println("Computing overlaps...")
    ratios, ratio_squares = compute_batch_overlap_py(
        selected_electrons,
        checkpoint.params_py,
        system_config, network_config,
        config_path;
        use_parallel=use_parallel
    )

    # for (i, r) in enumerate(ratios)
    #     println("Loop $(i-1)")
    #     println("ratio = $r, |ratio|^2 = $(abs(r)^2), overlap = $(abs(r)^2 / mean(abs(ratios).^2))")
    #     if i >= min(10, length(ratios)); break; end
    # end

    for i in eachindex(ratios)
        mean_ratio_i = mean(ratios[1:i])
        mean_ratio_sq_i = mean(ratio_squares[1:i])
        overlap_i = abs2(mean_ratio_i) / mean_ratio_sq_i
    
        println("Loop $(i-1)")
        println("ratio = $(ratios[i]), |ratio|^2 = $(ratio_squares[i])")
        println("overlap")
        println(overlap_i)
    
        # remove the next line if you want all steps, not just first 10
        if i >= min(10, length(ratios)); break; end
    end

    println("Any NaN in ratios? ", any(isnan, ratios))
    println("Any NaN in |ratio|^2? ", any(isnan, ratio_squares))

    valid_ratios = filter(!isnan, ratios)
    valid_ratio_squares = filter(!isnan, ratio_squares)

    if !isempty(valid_ratios) && !isempty(valid_ratio_squares)
        mean_ratio = mean(valid_ratios)
        mean_ratio_square = mean(valid_ratio_squares)
        overlap = abs2(mean_ratio) / mean_ratio_square
    else
        overlap = NaN
        mean_ratio = NaN + 0im
        mean_ratio_square = NaN
    end

    println("\nFinal Results:")
    println("Mean ratio: $mean_ratio")
    println("Mean |ratio|²: $mean_ratio_square")
    println("overlap")
    println(overlap)

    return overlap, ratios, ratio_squares
end

