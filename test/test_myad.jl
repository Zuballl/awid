using Test
using MyProject.MyAD
using MyProject.MyAD: sin, cos
using LinearAlgebra

# Test basic operations
@testset "Basic Operations" begin
    x = Dual(2.0, 1.0)
    y = Dual(3.0, 0.0)
    
    # Test addition
    @test x + y == Dual(5.0, 1.0)
    @test x + 2.0 == Dual(4.0, 1.0)
    @test 2.0 + x == Dual(4.0, 1.0)
    
    # Test multiplication
    @test x * y == Dual(6.0, 3.0)
    @test x * 2.0 == Dual(4.0, 2.0)
    @test 2.0 * x == Dual(4.0, 2.0)
end

# Test activation functions
@testset "Activation Functions" begin
    x = Dual(2.0, 1.0)
    
    # Test ReLU
    @test relu(x) == Dual(2.0, 1.0)
    @test relu(Dual(-2.0, 1.0)) == Dual(0.0, 0.0)
    
    # Test Sigmoid
    sigmoid_x = sigmoid(x)
    @test isapprox(sigmoid_x.val, 0.8807971, atol=1e-6)
    @test isapprox(sigmoid_x.grad, 0.1049936, atol=1e-6)
    
    # Test Tanh
    tanh_x = my_tanh(x)
    @test isapprox(tanh_x.val, 0.9640276, atol=1e-6)
    @test isapprox(tanh_x.grad, 0.0706508, atol=1e-6)
    
    # Test vectorized operations
    x_vec = [Dual(1.0, 1.0), Dual(2.0, 1.0)]
    @test all(relu.(x_vec) .== [Dual(1.0, 1.0), Dual(2.0, 1.0)])
    @test all(relu.([Dual(-1.0, 1.0), Dual(-2.0, 1.0)]) .== [Dual(0.0, 0.0), Dual(0.0, 0.0)])
end

# Test gradient computation
@testset "Gradient Computation" begin
    # Test simple function
    f(x) = x^2
    @test gradient(f, 2.0) ≈ 4.0
    
    # Test composite function
    g(x) = sin(x^2)
    @test gradient(g, 2.0) ≈ 4.0 * Base.cos(4.0)
    
    # Test vectorized gradient
    h(x) = sum(x.^2)
    @test gradient(h, [1.0, 2.0, 3.0]) ≈ [2.0, 4.0, 6.0]
end 