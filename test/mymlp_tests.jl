using Test
using LinearAlgebra
using Statistics: mean, std

include(joinpath(@__DIR__, "..", "src", "MyProject.jl"))
using .MyProject: MyMLP
using .MyMLP: Dense, ReLU, Softmax, Chain, Adam, train!, predict, load_data, train_test_split, forward

@testset "Layer Tests" begin
    # Test Dense layer initialization
    dense = Dense(4, 8)
    @test size(dense.weights) == (8, 4)
    @test size(dense.bias) == (8,)
    
    # Test ReLU forward pass
    relu = ReLU()
    x = MyMLP.MLPVariable([-1.0 0.0 1.0; -2.0 2.0 3.0])
    y = forward(relu, x)
    @test all(y.output .== [0.0 0.0 1.0; 0.0 2.0 3.0])
    
    # Test Softmax forward pass
    softmax = Softmax()
    x = MyMLP.MLPVariable([1.0 2.0; 3.0 4.0])
    y = forward(softmax, x)
    @test all(abs.(sum(y.output, dims=1) .- 1.0) .< 1e-10)
    @test all(y.output .> 0)
end

@testset "Model Tests" begin
    # Test Chain initialization
    model = Chain(
        Dense(4, 8),
        ReLU(),
        Dense(8, 3),
        Softmax()
    )
    @test length(model.layers) == 4
    
    # Test forward pass
    x = MyMLP.MLPVariable(randn(4, 10))
    y = model(x)
    @test size(y.output) == (3, 10)
    @test all(abs.(sum(y.output, dims=1) .- 1.0) .< 1e-10) 
    
    # Test predict function
    X = randn(4, 10)
    y_pred = predict(model, X)
    @test length(y_pred) == 10
    @test all(1 .<= y_pred .<= 3) 
end

@testset "Training Tests" begin
    X, Y = load_data()
    X_mean = mean(X, dims=1)
    X_std = std(X, dims=1)
    X = (X .- X_mean) ./ (X_std .+ 1e-8)
    
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y; test_ratio=0.2, seed=42)
    X_train = Matrix(X_train')
    X_test = Matrix(X_test')
    Y_train = Matrix(Y_train')
    Y_test = Matrix(Y_test')
    
    model = Chain(
        Dense(4, 8),
        ReLU(),
        Dense(8, 3),
        Softmax()
    )
    opt = Adam(0.01)
    
    # Test that training runs without errors
    @test_nowarn train!(model, X_train, Y_train, opt; epochs=1, batchsize=16)
    
   
    y_pred = predict(model, X_test)
    y_true = [argmax(Y_test[:,i]) for i in 1:size(Y_test,2)]
    accuracy = mean(y_pred .== y_true)
    @test accuracy > 0.0  
end



@testset "Data Loading Tests" begin
    # Test data loading
    X, Y = load_data()
    @test size(X) == (150, 4) 
    @test size(Y) == (150, 3)
    
    # Test one-hot encoding
    @test all(sum(Y, dims=2) .== 1) 
    @test all(Y .>= 0) 
    
    # Test train-test split
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y; test_ratio=0.2, seed=42)
    @test size(X_train, 1) == 120  
    @test size(X_test, 1) == 30   
end
