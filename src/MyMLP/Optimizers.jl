__precompile__(false)

using ..AD: MLPVariable

export SGD, Momentum, Adam, update!

mutable struct SGD
    learning_rate::Float64
end

function update!(opt::SGD, params::Vector{MLPVariable})
    for p in params
        if p.grad !== nothing
            p.output .-= opt.learning_rate .* p.grad
            p.grad = nothing
        end
    end
end

mutable struct Momentum
    learning_rate::Float64
    momentum::Float64
    velocity::Dict{MLPVariable,Matrix{Float64}}
end

Momentum(learning_rate::Float64, momentum::Float64) = Momentum(learning_rate, momentum, Dict{MLPVariable,Matrix{Float64}}())

function update!(opt::Momentum, params::Vector{MLPVariable})
    for p in params
        if p.grad !== nothing
            if !haskey(opt.velocity, p)
                opt.velocity[p] = zeros(size(p.output))
            end
            opt.velocity[p] = opt.momentum .* opt.velocity[p] .- opt.learning_rate .* p.grad
            p.output .+= opt.velocity[p]
            p.grad = nothing
        end
    end
end

mutable struct Adam
    learning_rate::Float64
    beta1::Float64
    beta2::Float64
    epsilon::Float64
    m::Dict{MLPVariable,Matrix{Float64}}
    v::Dict{MLPVariable,Matrix{Float64}}
    t::Int
end

Adam(learning_rate::Float64, beta1::Float64=0.9, beta2::Float64=0.999, epsilon::Float64=1e-8) = 
    Adam(learning_rate, beta1, beta2, epsilon, Dict{MLPVariable,Matrix{Float64}}(), Dict{MLPVariable,Matrix{Float64}}(), 0)

function update!(opt::Adam, params::Vector{MLPVariable})
    opt.t += 1
    for p in params
        if p.grad !== nothing
            if !haskey(opt.m, p)
                opt.m[p] = zeros(size(p.output))
                opt.v[p] = zeros(size(p.output))
            end
            
            opt.m[p] = opt.beta1 .* opt.m[p] .+ (1 - opt.beta1) .* p.grad
            
            opt.v[p] = opt.beta2 .* opt.v[p] .+ (1 - opt.beta2) .* (p.grad .^ 2)
            
            m̂ = opt.m[p] ./ (1 - opt.beta1^opt.t)
            
            v̂ = opt.v[p] ./ (1 - opt.beta2^opt.t)
            
            p.output .-= opt.learning_rate .* m̂ ./ (sqrt.(v̂) .+ opt.epsilon)
            p.grad = nothing
        end
    end
end