using Test
using MyAD

@testset "MyCNN Tests" begin
    @testset "CNN Network Forward" begin
        network = CNN()
        input = rand(Float64, 2, 1, 10, 10)  # batch=2, channels=1, height=10, width=10
        input32 = Float32.(input)
        output = forward(network, input32)
        @test size(output, 1) == 10  # 10 output classes (default in CNN)
    end
end 