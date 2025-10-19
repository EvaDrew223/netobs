"""
Overlap calculation between Psiformer and Laughlin wavefunctions
"""

using LinearAlgebra
using Statistics
using PyCall

include("data_loader_pycall.jl")
include("psiformer.jl") 
include("laughlin.jl")
include("math_utils.jl")

"""
Simple test version of overlap calculation - runs a minimal test
"""
function calculate_overlap_simple(checkpoint_path::String, config_path::String; steps::Int=5)
    # Load configuration and checkpoint
    system_config, network_config = load_config(config_path)
    checkpoint = load_checkpoint_pycall(checkpoint_path)
    
    println("✓ Config and checkpoint loaded successfully")
    println("  System: n_electrons=$(system_config.nspins), flux=$(system_config.flux)")
    println("  Network: type=$(network_config.type), layers=$(network_config.psiformer_num_layers)")
    println("  Using $(min(steps, size(checkpoint.data, 1))) electron configurations")
    
    # Run the actual overlap calculation with reduced steps
    overlap, all_ratios, all_ratio_squares = calculate_overlap(
        checkpoint_path, config_path; steps=steps, use_parallel=false)
    
    return overlap
end

"""
Compute overlap between neural network and Laughlin wavefunction for a batch of configurations
"""
function compute_batch_overlap(electrons_batch::Array{Float32, 3}, 
                              params_py::PyObject,
                              system_config, network_config,
                              config_path::String;
                              use_parallel::Bool=true)
    
    batch_size, n_electrons, n_dim = size(electrons_batch)
    
    # First, compute all log ratios to calculate the shift (matching Python version)
    log_ratios = zeros(ComplexF64, batch_size)
    
    if use_parallel && nprocs() > 1
        # Parallel computation across processes
        log_ratios = pmap(1:batch_size) do i
            electron_config = electrons_batch[i, :, :]
            compute_log_ratio(electron_config, params_py, system_config, network_config, config_path)
        end
    else
        # Sequential computation
        for i in 1:batch_size
            electron_config = electrons_batch[i, :, :]
            log_ratios[i] = compute_log_ratio(electron_config, params_py, system_config, network_config, config_path)
        end
    end
    
    # Calculate shift for numerical stability (matching Python version)
    shift = mean(log_ratios)
    println("Debug: shift = $shift")
    
    # Compute final ratios with shift applied
    ratios = exp.(log_ratios .- shift)
    ratio_squares = abs.(ratios).^2
    
    return ratios, ratio_squares
end

"""
Compute log ratio (logphi - logpsi) for a single electron configuration
"""
function compute_log_ratio(electrons::Array{Float32, 2}, 
                          params_py::PyObject,
                          system_config, network_config,
                          config_path::String)
    
    # Compute log wavefunction from trained neural network (Psiformer)
    logpsi = psiformer_forward(electrons, params_py, config_path)
    println("Debug: logpsi = $logpsi")
    
    # Compute log wavefunction from Laughlin reference state  
    logphi = laughlin_forward(electrons, system_config)
    println("Debug: logphi = $logphi")
    
    # Return log ratio
    log_ratio = logphi - logpsi
    println("Debug: log_ratio = $log_ratio")
    
    return log_ratio
end

"""
Compute overlap for a single electron configuration (legacy function - kept for compatibility)
"""
function compute_single_overlap(electrons::Array{Float32, 2}, 
                               params::Dict{String, Any},
                               system_config, network_config)
    
    log_ratio = compute_log_ratio(electrons, params, system_config, network_config)
    ratio = exp(log_ratio)
    println("Debug: ratio = $ratio")
    ratio_square = abs(ratio)^2
    println("Debug: ratio_square = $ratio_square")
    
    return ratio, ratio_square
end

"""
Main overlap calculation function matching Python version
"""
function calculate_overlap(checkpoint_path::String, config_path::String; 
                          steps::Int=50, use_parallel::Bool=true)
    
    println("Loading configuration from: $config_path")
    system_config, network_config = load_config(config_path)
    
    println("Loading checkpoint from: $checkpoint_path")
    checkpoint = load_checkpoint_pycall(checkpoint_path)
    
    println("System config:")
    println("  Flux: $(system_config.flux)")
    println("  Number of spins: $(system_config.nspins)")
    println("  Interaction: $(system_config.interaction_type)")
    
    println("Network config:")
    println("  Type: $(network_config.type)")
    println("  Heads: $(network_config.psiformer_num_heads)")
    println("  Layers: $(network_config.psiformer_num_layers)")
    println("  Head dim: $(network_config.psiformer_heads_dim)")
    
    # Get the electron configurations from checkpoint
    all_electrons = checkpoint.data  # Shape: (batch_size, n_electrons, 2)
    total_samples = size(all_electrons, 1)
    
    println("Total available samples: $total_samples")
    println("Using $steps steps for overlap calculation")
    
    # Select subset of samples for overlap calculation
    if steps > total_samples
        println("Warning: Requested steps ($steps) > available samples ($total_samples)")
        steps = total_samples
    end

    # Use evenly spaced samples
    sample_indices = steps == 1 ? [1] : round.(Int, range(1, total_samples, length=steps))
    selected_electrons = all_electrons[sample_indices, :, :]

    println("Computing overlaps...")

    # Compute ratios for the whole batch at once (shift over the batch)
    ratios, ratio_squares = compute_batch_overlap(
        selected_electrons,
        checkpoint.params_py,
        system_config, network_config, config_path;
        use_parallel=use_parallel
    )

    # Optional: print a few entries
    for (i, r) in enumerate(ratios)
        println("Loop $(i-1)")
        println("ratio = $r, |ratio|^2 = $(abs(r)^2)")
        if i >= min(10, length(ratios)); break; end
    end

    # NaN checks
    println("Any NaN in ratios? ", any(isnan, ratios))
    println("Any NaN in |ratio|^2? ", any(isnan, ratio_squares))

    # Aggregate exactly like Python digest()
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

