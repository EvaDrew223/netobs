## Mathematical utility functions used by laughlin.jl

using LinearAlgebra
using SpecialFunctions
using Statistics

"""
Compute signed log determinant similar to JAX's slogdet
Returns (sign, logdet) tuple
"""
function slogdet(A::AbstractMatrix)
    val, sign = logabsdet(A)
    return sign, val
end

"""
Compute log-sum-exp trick for numerical stability
"""
function logsumexp(x::AbstractVector)
    max_x = maximum(x)
    return log(sum(exp.(x .- max_x))) + max_x
end

"""
Compute softmax with numerical stability
"""
function softmax(x::AbstractVector)
    max_x = maximum(x)
    exp_x = exp.(x .- max_x)
    return exp_x ./ sum(exp_x)
end

"""
Layer normalization
"""
function layer_norm(x::AbstractArray; ε::Float64=1e-5)
    μ = mean(x, dims=length(size(x)))
    σ² = var(x, dims=length(size(x)), corrected=false)
    return (x .- μ) ./ sqrt.(σ² .+ ε)
end

"""
Compute binomial coefficient
"""
function binomial_coeff(n::Real, k::Real)
    return exp(loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1))
end

"""
Multi-head attention mechanism
"""
function multi_head_attention(x::AbstractMatrix, 
                              W_q::AbstractArray, b_q::AbstractVector,
                              W_k::AbstractArray, b_k::AbstractVector, 
                              W_v::AbstractArray, b_v::AbstractVector,
                              W_o::AbstractArray, b_o::AbstractVector,
                              num_heads::Int, head_dim::Int)
    seq_len, d_model = size(x)

    Q = reshape(x * W_q .+ b_q', seq_len, num_heads, head_dim)
    K = reshape(x * W_k .+ b_k', seq_len, num_heads, head_dim)
    V = reshape(x * W_v .+ b_v', seq_len, num_heads, head_dim)

    out = zeros(eltype(Q), seq_len, num_heads, head_dim)
    scale = 1 / sqrt(head_dim)
    for h in 1:num_heads
        Qh = Q[:, h, :]                # (L, D)
        Kh = K[:, h, :]                # (L, D)
        Vh = V[:, h, :]                # (L, D)
        scores = (Qh * Kh') .* scale   # (L, L)
        # row-wise softmax
        attn = mapslices(softmax, scores; dims=2)
        out[:, h, :] = attn * Vh       # (L, D)
    end
    concat = reshape(out, seq_len, num_heads * head_dim)
    return concat * W_o .+ b_o'
end

"""
Dense/fully connected layer
"""
function dense_layer(x::AbstractArray, W::AbstractMatrix, b::Union{AbstractVector, Nothing}=nothing)
    result = x * W
    if b !== nothing
        result = result .+ b'
    end
    return result
end

"""
Apply activation function
"""
function apply_activation(x::AbstractArray, activation::String)
    if activation == "tanh"
        return tanh.(x)
    elseif activation == "relu"
        return max.(0, x)
    elseif activation == "gelu"
        return x .* (0.5 * (1.0 .+ tanh.(sqrt(2/π) * (x .+ 0.044715 * x.^3))))
    else
        return x  # linear/identity
    end
end