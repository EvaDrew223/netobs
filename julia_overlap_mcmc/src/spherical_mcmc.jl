## Contains spherical MCMC proposal and Metropolis-Hastings logic

module SphericalMCMC

using Random
using LinearAlgebra
using PyCall

export sph_sampling_batch, logprob_batch, mh_step_batch

# Batched spherical proposal: local Gaussian tilt (atan(N(0,1)*width)) and uniform azimuth
function sph_sampling_batch(e::Array{Float32,3}, width::Float32, rng::AbstractRNG)
    B, N, _ = size(e)
    θ = @view e[:, :, 1]
    φ = @view e[:, :, 2]

    θp = atan.(randn(rng, Float32, B, N) .* width)
    φp = 2f0 * Float32(pi) .* rand(rng, Float32, B, N)

    sθp = sin.(θp); cθp = cos.(θp)
    cφp = cos.(φp); sφp = sin.(φp)
    x′ = sθp .* cφp
    y′ = sθp .* sφp
    z′ = cθp

    cφ = cos.(φ); sφ = sin.(φ)
    cθ = cos.(θ); sθ = sin.(θ)

    # x2 = Rz(φ)*Ry(θ) * [x′,y′,z′]
    x2 = cφ .* (cθ .* x′ .+ sθ .* z′) .- sφ .* y′
    y2 = sφ .* (cθ .* x′ .+ sθ .* z′) .+ cφ .* y′
    z2 = - sθ .* x′ .+ cθ .* z′

    θ2 = acos.(clamp.(z2, -1f0, 1f0))
    denom = max.(sin.(θ2), 1f-7)  # guard division by 0
    φ2 = sign.(y2) .* acos.(clamp.(x2 ./ denom, -1f0, 1f0))

    e2 = similar(e)
    @views e2[:, :, 1] .= θ2
    @views e2[:, :, 2] .= φ2
    return e2
end

# Batched log-probability: 2*Re(log ψNN) using Python Psiformer batched forward
function logprob_batch(e::Array{Float32,3}, params_py::PyObject, config_path::String)
    # Import batched forward from psiformer.jl (the module that defines it must be included by caller)
    @assert isdefined(Main, :psiformer_forward_batch) "psiformer_forward_batch not found. Include psiformer.jl first."
    logpsi = Main.psiformer_forward_batch(e, params_py, config_path)  # Vector{ComplexF64}
    return 2 .* real.(logpsi)  # Vector{Float64}
end

# Repeat Metropolis-Hastings updates for the whole batch
function mh_step_batch(e::Array{Float32,3}, params_py::PyObject, config_path::String,
                       width::Float32, n_moves::Int, rng::AbstractRNG)
    lp = logprob_batch(e, params_py, config_path)  # (B,)
    for _ in 1:n_moves
        e2 = sph_sampling_batch(e, width, rng)
        lp2 = logprob_batch(e2, params_py, config_path)
        # Accept if Δlogp > log U
        acc = lp2 .- lp .> log.(rand(rng, Float64, size(lp)))
        @inbounds for b in 1:size(e,1)
            if acc[b]
                @views e[b, :, :] .= e2[b, :, :]
                lp[b] = lp2[b]
            end
        end
    end
    return e
end

end # module