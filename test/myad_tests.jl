using Test

include(joinpath(@__DIR__, "..", "src", "MyProject.jl"))
using .MyProject: AD
using .AD: Variable, topological_sort, forward!, backward!, gradient

# 1) ScalarOp basic AD tests
@testset "ScalarOp basic AD tests" begin
    x = Variable(2.0; name="x")
    y = Variable(3.0; name="y")
    z = x + y
    g = topological_sort(z)
    forward!(g)
    @test z.output == 5.0
    backward!(g)
    @test x.gradient == 1.0 && y.gradient == 1.0

    x = Variable(4.0; name="x")
    y = Variable(5.0; name="y")
    z = x * y
    g = topological_sort(z); forward!(g); backward!(g)
    @test z.output == 20.0
    @test x.gradient == 5.0 && y.gradient == 4.0

    x = Variable(3.0; name="x")
    y = Variable(2.0; name="y")
    z = x ^ y
    g = topological_sort(z); forward!(g); backward!(g)
    @test z.output == 9.0
    @test isapprox(x.gradient, 2 * 3.0^(2-1); atol=1e-8)
    @test isapprox(y.gradient, log(3.0) * 3.0^2; atol=1e-8)

    x = Variable(pi/2; name="x")
    z = sin(x)
    g = topological_sort(z); forward!(g); backward!(g)
    @test isapprox(z.output, 1.0; atol=1e-8)
    @test isapprox(x.gradient, cos(pi/2); atol=1e-8)
end

# 2) BroadcastedOp tests
@testset "BroadcastedOp tests" begin
    x = Variable([1.0, 2.0, 3.0]; name="x")
    z = exp.(x)
    g = topological_sort(z)
    forward!(g)
    @test all(z.output .== exp.([1.0, 2.0, 3.0]))
    backward!(g)
    @test all(x.gradient .== exp.([1.0, 2.0, 3.0]))

    x = Variable([1.0, 2.0, 3.0]; name="x")
    z2 = log.(x)
    g2 = topological_sort(z2)
    forward!(g2)
    @test all(z2.output .== log.([1.0, 2.0, 3.0]))
    backward!(g2)
    @test all(x.gradient .== 1.0 ./ [1.0, 2.0, 3.0])
end
