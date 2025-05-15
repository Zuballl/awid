export load_data, train_test_split

include(joinpath(@__DIR__, "..", "..", "iris.jl"))

using Random

function load_data()
    return inputs, targets
end

function train_test_split(X::Matrix{Float64}, y::Matrix{Float64}; test_ratio=0.2, seed=123)
    rng = MersenneTwister(seed)
    n = size(X, 1)
    idx = shuffle(rng, 1:n)
    n_test = Int(round(n * test_ratio))
    test_idx = idx[1:n_test]
    train_idx = idx[n_test+1:end]
    return X[train_idx, :], y[train_idx, :], X[test_idx, :], y[test_idx, :]
end