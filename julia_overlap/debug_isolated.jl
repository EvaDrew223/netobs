using PyCall
using LinearAlgebra
using Statistics

include("src/data_loader_pycall.jl")
include("src/laughlin.jl")
include("src/psiformer.jl")

function test_components_separately()
    checkpoint_path = "/Users/jihangzhu/Documents/netobs/DeepHall_local/data/seed_1758632847/tillicum/DeepHall_n8l21_optimal/ckpt_199999.npz"
    config_path = "/Users/jihangzhu/Documents/netobs/DeepHall_local/data/seed_1758632847/tillicum/DeepHall_n8l21_optimal/config.yml"
    
    println("=== Testing components separately ===")
    
    system_config, network_config = load_config(config_path)
    checkpoint = load_checkpoint_pycall(checkpoint_path)
    
    println("System: flux=$(system_config.flux), nspins=$(system_config.nspins)")
    
    # Use just first electron configuration
    electrons = checkpoint.data[1, :, :]  # First configuration, all 8 electrons
    println("Test electrons shape: $(size(electrons))")
    println("First few electron positions:")
    for i in 1:min(3, size(electrons, 1))
        println("  Electron $i: $(electrons[i, :])")
    end
    
    # Test 1: Laughlin calculation
    println("\n=== Test 1: Laughlin wavefunction ===")
    try
        logphi = laughlin_forward(electrons, system_config)
        
        println("SUCCESS: Laughlin calculation")
        println("  logphi = $logphi")
        println("  Is finite? $(isfinite(logphi))")
        println("  Is NaN? $(isnan(logphi))")
        println("  Real part: $(real(logphi))")  
        println("  Imag part: $(imag(logphi))")
        
    catch e
        println("ERROR in Laughlin calculation: $e")
        return false
    end
    
    # Test 2: Simple Psiformer (just parameter access)
    println("\n=== Test 2: Psiformer parameter access ===")
    try
        params = checkpoint.params
        println("Parameter keys at root level:")
        for key in keys(params)
            println("  $key")
        end
        
        # Test accessing a specific parameter
        if haskey(params, "params")
            nested_params = params["params"]
            println("\nNested parameter keys:")
            for key in keys(nested_params)
                println("  $key")
            end
            
            # Try to access a specific layer
            if haskey(nested_params, "PsiformerLayers_0")
                layer0 = nested_params["PsiformerLayers_0"]
                println("\nPsiformerLayers_0 keys:")
                for key in keys(layer0)
                    println("  $key")
                end
            end
        end
        
        println("SUCCESS: Parameter access works")
        
    catch e
        println("ERROR in parameter access: $e")
        return false
    end
    
    # Test 3: Try minimal Psiformer forward
    println("\n=== Test 3: Minimal Psiformer forward ===")
    try
        logpsi = psiformer_forward(electrons, checkpoint.params, system_config, network_config)
        
        println("Psiformer result:")
        println("  logpsi = $logpsi")
        println("  Is finite? $(isfinite(logpsi))")
        println("  Is NaN? $(isnan(logpsi))")
        println("  Real part: $(real(logpsi))")  
        println("  Imag part: $(imag(logpsi))")
        
        if isnan(logpsi)
            println("ERROR: Psiformer forward produced NaN!")
            return false
        else
            println("SUCCESS: Psiformer forward works")
        end
        
    catch e
        println("ERROR in Psiformer forward: $e")
        println("Stacktrace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        return false
    end
    
    return true
end

# Run test
test_components_separately()