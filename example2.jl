using JuMP, Ipopt, LinearAlgebra

model = Model(Ipopt.Optimizer)

@variable(model, x[1:4])

@objective(model, Min, x[2]^2 + x[3]^2 + exp(x[1]^2) + x[4]^2 - 3*x[4] + exp(x[2] - x[3]))

@constraint(model, 2*x[1] + x[2] + x[3] + 4*x[4] == 7)
@constraint(model, x[1] + x[2] + x[3] + x[4] == 5)

optimize!(model)

println("status = ", termination_status(model))
println("x = ", value.(x))
println("obj = ", objective_value(model))
