#!/usr/bin/env julia

## Main CLI interface and argument parsing for Julia overlap calculation (with MCMC parity)

using ArgParse
using Printf
using Distributed
using YAML, NPZ, LinearAlgebra, Statistics, SpecialFunctions
using Random

include("overlap_calculator_py.jl")
include("overlap_calculator_py_mcmc.jl")  # NEW

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
            help = "Number of measurement steps"
            arg_type = Int
            default = 5

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

        "--mcmc-steps"
            help = "Number of MH updates between measurement steps"
            arg_type = Int
            default = 10

        "--mcmc-burn-in"
            help = "Burn-in factor (burn-in = burn_in * mcmc_steps)"
            arg_type = Int
            default = 100

        "--seed"
            help = "RNG seed"
            arg_type = Int

        "--no-mcmc"
            help = "Disable MCMC (single-shot evaluation semantics)"
            action = :store_true
    end

    return parse_args(s)
end

"""
Main function
"""
function main()
    parsed_args = parse_commandline()

    checkpoint_path = parsed_args["checkpoint"]
    config_path = parsed_args["config"]
    steps = parsed_args["steps"]
    use_parallel = parsed_args["parallel"]
    simple_mode = parsed_args["simple"]
    verbose = parsed_args["verbose"]
    pyexact = parsed_args["pyexact"]
    mcmc_steps = get(parsed_args, "mcmc-steps", 10)
    mcmc_burn_in = get(parsed_args, "mcmc-burn-in", 100)
    seed = get(parsed_args, "seed", nothing)
    no_mcmc = parsed_args["no-mcmc"]

    if use_parallel && nprocs() == 1
        addprocs(Sys.CPU_THREADS ÷ 2)
        @everywhere begin
            include("overlap_calculator.jl")
            include("overlap_calculator_py.jl")
            include("overlap_calculator_py_mcmc.jl")
        end
    end

    if simple_mode
        test_steps = min(steps, 5)
        overlap = calculate_overlap_simple(checkpoint_path, config_path, steps=test_steps)
        @printf("Final overlap: %.6e\n", overlap)
        return
    end

    if pyexact
        if no_mcmc
            overlap, ratios, ratio_squares = calculate_overlap_py(
                checkpoint_path, config_path; steps=steps, use_parallel=use_parallel
            )
        else
            overlap, r_t, q_t = calculate_overlap_py_mcmc(
                checkpoint_path, config_path;
                steps=steps, mcmc_steps=mcmc_steps, mcmc_burn_in=mcmc_burn_in,
                seed=seed, use_parallel=use_parallel
            )
        end
    else
        # legacy (Julia-native Laughlin); not recommended for strict parity
        overlap, ratios, ratio_squares = calculate_overlap(
            checkpoint_path, config_path; steps=steps, use_parallel=use_parallel
        )
    end

    println("\n=== RESULTS ===")
    @printf("Final overlap: %.6e\n", overlap)
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
    main()
end