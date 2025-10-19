#!/usr/bin/env julia

"""
Main CLI interface for Julia overlap calculation
"""

using ArgParse
using Printf
using Distributed
using YAML, NPZ, LinearAlgebra, Statistics, SpecialFunctions

include("overlap_calculator.jl")
include("overlap_calculator_py.jl")

"""
Parse command line arguments
"""
function parse_commandline()
    s = ArgParseSettings()
    
    @add_arg_table s begin
        "--checkpoint", "-c"
            help = "Path to the checkpoint (.npz) file"
            required = true
            
        "--config", "-f"  
            help = "Path to the config (.yml) file"
            required = true
            
        "--steps", "-s"
            help = "Number of steps for overlap calculation"
            arg_type = Int
            default = 50
            
        "--parallel", "-p"
            help = "Use parallel computation"
            action = :store_true
            
        "--simple"
            help = "Use simplified testing mode"
            action = :store_true
            
        "--verbose", "-v"
            help = "Verbose output"
            action = :store_true

        "--pyexact"
            help = "Use Python-backed exact-parity networks (Psiformer + Laughlin)"
            action = :store_true
    end
    
    return parse_args(s)
end

"""
Main function
"""
function main()
    println("=== Julia Overlap Calculator ===")
    println("Translating Python DeepHall overlap calculation to Julia")
    println()
    
    # Parse arguments
    parsed_args = parse_commandline()
    
    checkpoint_path = parsed_args["checkpoint"]
    config_path = parsed_args["config"]
    steps = parsed_args["steps"]
    use_parallel = parsed_args["parallel"]
    simple_mode = parsed_args["simple"]
    verbose = parsed_args["verbose"]
    pyexact = parsed_args["pyexact"]
    
    # Validate file paths
    if !isfile(checkpoint_path)
        println("Error: Checkpoint file not found: $checkpoint_path")
        exit(1)
    end
    
    if !isfile(config_path)
        println("Error: Config file not found: $config_path")
        exit(1)
    end
    
    if verbose
        println("Arguments:")
        println("  Checkpoint: $checkpoint_path")
        println("  Config: $config_path")
        println("  Steps: $steps")
        println("  Parallel: $use_parallel")
        println("  Simple mode: $simple_mode")
        println("  Pyexact: $pyexact")
        println()
    end
    
    # Add worker processes for parallel computation
    if use_parallel
        if nprocs() == 1
            println("Adding worker processes for parallel computation...")
            addprocs(Sys.CPU_THREADS ÷ 2)  # Use half the available cores
            println("Total processes: $(nprocs())")
            
            # Load modules on all workers
            @everywhere begin
                include("overlap_calculator.jl")
                include("overlap_calculator_py.jl")
            end
        end
    end
    
    try
        println("Starting overlap calculation...")
        start_time = time()
        
        if simple_mode
            println("Running in simple test mode...")
            test_steps = min(steps, 5)
            overlap = calculate_overlap_simple(checkpoint_path, config_path, steps=test_steps)
        else
            if pyexact
                overlap, all_ratios, all_ratio_squares = calculate_overlap_py(
                    checkpoint_path, config_path;
                    steps=steps, use_parallel=use_parallel
                )
            else
                overlap, all_ratios, all_ratio_squares = calculate_overlap(
                    checkpoint_path, config_path;
                    steps=steps, use_parallel=use_parallel
                )
            end
        end
        
        end_time = time()
        elapsed_time = end_time - start_time
        
        println()
        println("=== RESULTS ===")
        @printf("Final overlap: %.6e\n", overlap)
        @printf("Computation time: %.2f seconds\n", elapsed_time)
        
        if !simple_mode
            # Additional statistics
            valid_ratios = filter(!isnan, all_ratios)
            valid_ratio_squares = filter(!isnan, all_ratio_squares)
            
            if length(valid_ratios) > 0
                println()
                println("Statistics:")
                @printf("Valid samples: %d/%d\n", length(valid_ratios), steps)
                @printf("Mean |ratio|: %.6e\n", mean(abs.(valid_ratios)))
                @printf("Std |ratio|: %.6e\n", std(abs.(valid_ratios)))
                @printf("Mean |ratio|²: %.6e\n", mean(valid_ratio_squares))
                @printf("Std |ratio|²: %.6e\n", std(valid_ratio_squares))
            end
        end
        
    catch e
        println("Error during calculation: $e")
        if verbose
            println("Stack trace:")
            for (exc, bt) in Base.catch_stack()
                showerror(stdout, exc, bt)
                println()
            end
        end
        exit(1)
    finally
        # Clean up worker processes
        if use_parallel && nprocs() > 1
            rmprocs(workers())
        end
    end
end

"""
Simple test function to verify the setup
"""
function test_setup()
    println("Testing Julia environment setup...")
    
    try
        println("✓ All required packages loaded successfully")
        
        # Test basic math operations
        test_matrix = ComplexF64[1+1im 2; 3 4+2im]
        sign, logdet = slogdet(test_matrix)
        println("✓ Linear algebra operations working")
        println("✓ Setup verification complete")
        
    catch e
        println("✗ Setup verification failed: $e")
        return false
    end
    
    return true
end

# Run main function if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) > 0 && ARGS[1] == "test"
        test_setup()
    else
        main()
    end
end