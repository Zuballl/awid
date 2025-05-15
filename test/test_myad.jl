using Test
using MyAD

@testset "MyAD Tests" begin
    @testset "Basic Operations" begin
        # Test addition
        x = Dual(2.0, 1.0)
        y = Dual(3.0, 0.0)
        z = x + y
        @test z.val ≈ 5.0
        @test z.grad ≈ 1.0

        # Test multiplication
        z = x * y
        @test z.val ≈ 6.0
        @test z.grad ≈ 3.0

        # Test division
        z = x / y
        @test z.val ≈ 2/3
        @test z.grad ≈ 1/3
    end

    @testset "Activation Functions" begin
        x = Dual(0.0, 1.0)
        
        # Test sigmoid
        s = sigmoid(x)
        @test s.val ≈ 0.5
        @test s.grad ≈ 0.25

        # Test tanh
        t = my_tanh(x)
        @test t.val ≈ 0.0
        @test t.grad ≈ 1.0

        # Test relu
        r = relu(x)
        @test r.val ≈ 0.0
        @test r.grad ≈ 0.0

        x = Dual(1.0, 1.0)
        r = relu(x)
        @test r.val ≈ 1.0
        @test r.grad ≈ 1.0
    end

    @testset "Gradient Computation" begin
        # Test simple function
        f(x) = x^2
        @test gradient(f, 2.0) ≈ 4.0

        # Test composite function
        f(x) = MyAD.sin(x^2)
        @test gradient(f, 1.0) ≈ 2.0 * Base.cos(1.0)

        # Test vectorized gradient
        f(x) = sum(x.^2)
        x = [1.0, 2.0, 3.0]
        grads = gradient(f, x)
        @test grads ≈ [2.0, 4.0, 6.0]
    end

    @testset "Neural Network Operations" begin
        # Test forward pass for Dense only
        layer = Dense(2, 3, sigmoid)
        input = [1.0 2.0; 3.0 4.0]  # 2 samples, 2 features
        output = forward(layer, input)
        @test size(output) == (3, 2)  # 3 outputs, 2 samples
    end
end 