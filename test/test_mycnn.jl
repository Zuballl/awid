using Test
using MyProject
using MyProject.MyCNN
using MyProject.MyCNN: forward as cnn_forward, backward as cnn_backward, NeuralNetwork
using MyProject.MyCNN: mse_loss, cross_entropy_loss
using MyProject.MyAD
using MyProject.MyAD: relu, sigmoid, tanh
using LinearAlgebra

# Test data
x = rand(Float32, 2, 1, 28, 28)  # 2 samples: (batch, channels, height, width)
y = [1, 2]  # Labels for the samples

# Test Conv2D layer
@testset "Conv2D Layer" begin
    # Test initialization
    conv = Conv2D((3, 3), 1, 3)  # 1 input channel, 3 output channels, 3x3 kernel
    @test size(conv.filters) == (3, 3, 1, 3)
    @test size(conv.bias) == (3,)
    
    # Test forward pass
    output = cnn_forward(conv, x).output
    @test size(output) == (2, 3, 26, 26)  # Output size after 3x3 convolution
    
    # Test backward pass
    grad = rand(Float32, size(output))
    cnn_backward(CNNVariable(x), grad)
    @test size(conv.filters_grad) == size(conv.filters)
    @test size(conv.bias_grad) == size(conv.bias)
end

# Test MaxPool2D layer
@testset "MaxPool2D Layer" begin
    # Test initialization
    pool = MaxPool2D((2, 2))  # 2x2 pooling
    
    # Test forward pass
    output = cnn_forward(pool, x).output
    @test size(output) == (2, 1, 14, 14)  # Output size after 2x2 pooling
    
    # Test backward pass
    grad = rand(Float32, size(output))
    var = CNNVariable(x)
    cnn_backward(var, grad)
    @test size(var.grad) == size(output)
end

# Test Dense layer
@testset "Dense Layer" begin
    # Test initialization
    dense = Dense(784, 10, relu)  # 784 input features, 10 output features
    
    # Test forward pass
    input = rand(Float64, 2, 784)  # 2 samples, batch first
    output = cnn_forward(dense, input)
    @test size(output) == (10, 2)
    
    # Test backward pass
    var = CNNVariable(input)
    grad = rand(Float32, size(output)..., 1, 1)  # Convert to 4D tensor
    cnn_backward(var, grad)
    @test size(dense.weights_grad) == size(dense.weights)
    @test size(dense.bias_grad) == size(dense.bias)
end

# Test activation functions
@testset "Activation Functions" begin
    x = rand(Float32, 10)
    
    # Test ReLU
    @test all(relu.(x) .>= 0)
    @test all(relu.(-x) .== 0)
    
    # Test Sigmoid
    sigmoid_output = sigmoid.(x)
    @test all(0 .<= sigmoid_output .<= 1)
    
    # Test Tanh
    tanh_output = tanh.(x)
    @test all(-1 .<= tanh_output .<= 1)
end

# Test NeuralNetwork
@testset "NeuralNetwork" begin
    # Test initialization
    layers = [
        Conv2D((3, 3), 1, 16),
        MaxPool2D((2, 2)),
        Conv2D((3, 3), 16, 32),
        MaxPool2D((2, 2)),
        Dense(32 * 5 * 5, 10, relu)
    ]
    model = NeuralNetwork(layers)
    
    # Test forward pass
    output = cnn_forward(model, x)
    @test size(output) == (10, 2)
    
    # Test backward pass
    grad = rand(Float32, size(output))
    input_grad = cnn_backward(model, grad)
    @test size(input_grad) == size(x)
end

# Test loss functions
@testset "Loss Functions" begin
    # Test MSE loss
    y_pred = CNNVariable(rand(Float32, 10, 2, 1, 1))
    y_true = rand(Float32, 10, 2, 1, 1)
    loss = mse_loss(y_pred, y_true)
    @test loss >= 0
    
    # Test cross entropy loss
    y_pred = CNNVariable(rand(Float32, 10, 2, 1, 1))
    y_true = rand(Float32, 10, 2, 1, 1)
    loss = cross_entropy_loss(y_pred, y_true)
    @test loss >= 0
end

# Test training loop
@testset "Training Loop" begin
    # Create a simple network
    layers = [
        Conv2D((3, 3), 1, 16),
        MaxPool2D((2, 2)),
        Dense(16 * 13 * 13, 10, relu)
    ]
    model = NeuralNetwork(layers)
    
    # Test one training step
    loss = train_step!(model, x, y)
    @test loss >= 0
end 