using DifferentialEquations, LaTeXStrings, LinearAlgebra, Plots, JuMP, Ipopt

# Parameters
L = 10
T = 400
Y₀ = 1
N = 1
Budget = 1
α = [1/3, 1/3, 1/3]
# β = [1.1,1.2,1.3] # superlinear scaling
β = [0.7,0.8,0.9] # sublinear scaling

# Determine allocations that maximize Nash social welfare
function Nash_allocation(u)
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    @variable(model, X[1:L^2, 1:3] >= 0)
    
    # Multibudget constraint
    # @constraint(model, sum(X[i, 1] for i in 1:L^2) == Budget)
    # @constraint(model, sum(X[i, 2] for i in 1:L^2) == Budget)
    # @constraint(model, sum(X[i, 3] for i in 1:L^2) == Budget)

    # Single budget constraint
    @constraint(model, sum(X[i, j] for i in 1:L^2, j in 1:3) == Budget)
    
    # Objective: maximize Nash social welfare
    @NLobjective(model, Max, prod((sum(α[j] * log(1.0 + X[i,j] / (Y₀*(u[i]*N)^β[j])) for j in 1:3))^(u[i]) for i in 1:L^2))
    
    # Solve the model
    optimize!(model)
    
    return value.(X)
end

# Calculate utility
function calc_utility(u,p)
    utility = zeros(L^2)
    allocation = Nash_allocation(u)
    for i=1:L
        for j=1:L
            for k=1:3
                utility[L*(i-1)+j] += p[1][k]*log(1.0 + allocation[L*(i-1)+j,k]/(Y₀*(u[L*(i-1)+j]*N)^p[2][k]))
            end
        end
    end
    return utility
end

# Replicator equations
function replicator!(du,u,p,t)
    utility = calc_utility(u,p)
    for i=1:L
        for j=1:L
            diff_utility = 0
            for m=1:L
                for n=1:L
                    if m!=i || n!=j
                        diff_utility += u[L*(m-1)+n]*(utility[L*(i-1)+j] - utility[L*(m-1)+n])/((i-m)^2 + (j-n)^2)
                    end
                end
            end
            du[L*(i-1)+j] = u[L*(i-1)+j]*diff_utility
        end
    end
end

# Initial conditions and parameters
u0 = rand(L^2)
u0 = u0/sum(u0)
p = (α,β)
tspan = (0.0,T)

# Solve differential equations
prob = ODEProblem(replicator!,u0,tspan,p)
sol = solve(prob,abstol=1e-8,reltol=1e-8,saveat=1)
# sol = solve(prob,saveat=1)

# Format data
population = vcat(hcat(u0, zeros(L^2)),zeros(Int(T)*(L^2),2))
social_welfare = vcat(hcat(calc_utility(u0,p), zeros(L^2)),zeros(Int(T)*(L^2),2))
global count = L^2
for j = 2:T+1
    for i = 1:L^2
        global count = count + 1
        social_welfare[count,:] = [calc_utility(sol[:,j],p)[i], j]
        population[count,:] = [sol[i,j], j]
    end
end
city_social_welfare = transpose(sum(reshape(social_welfare[:,1],L^2,T+1),dims=1)/L^2)
utilitarian_social_welfare = transpose(sum(reshape(social_welfare[:,1].*population[:,1],L^2,T+1),dims=1))
Nash_social_welfare = prod.(eachcol(reshape(social_welfare[:,1],L^2,T+1).^(reshape(population[:,1],L^2,T+1))))
Rawlsian_social_welfare = transpose(minimum(reshape(social_welfare[:,1],L^2,T+1),dims=1))
avg_population = transpose(sum(sol,dims=1)/L^2)
heat = reshape(sol[Int(200)],L,L)
lineSW = transpose(reshape(social_welfare[:,1],L^2,T+1))
linePop = transpose(reshape(population[:,1],L^2,T+1))

# Plot results
pyplot()

# Superlinear plotting
# p1 = heatmap(heat, title = L"Population at time $t=200$", xaxis=false, yaxis=false, colorbar_title=L"Population, $p_i$", c = cgrad([RGB(1,1,1),RGB(1,113/255,206/255)]), clim=(0.0,0.02001),aspect_ratio=:equal)
# p2 = plot(collect(0:T), lineSW, title= L"Utility $U_i$ vs time $t$", label=false, xlabel=L"Time, $t$", ylabel=L"Utility, $U_i$", color = :black, alpha=0.3)
# plot!(collect(0:T),city_social_welfare, label = L"Average city utility, $W_C$", linewidth=3, color=RGB(1,113/255,206/255))
# plot!(collect(0:T),utilitarian_social_welfare, label = L"Utilitarian, $W_U$", linewidth=3, color=RGB(1/255,205/255,254/255))
# plot!(collect(0:T),Nash_social_welfare, label = L"Nash, $W_N$", linewidth=3, color=RGB(5/255,1,161/255), linestyle=:dash)
# plot!(collect(0:T),Rawlsian_social_welfare, label = L"Rawlsian, $W_R$", linewidth=3, color=RGB(185/255,103/255,1),legend=:topright,ylims=(1.15,1.85))
# p3 = plot(collect(0:T), linePop, title= L"Population $p_i$ vs time $t$", label=false, xlabel=L"Time, $t$", ylabel=L"Population, $p_i$", color = :black, alpha=0.3)
# plot!(collect(0:T),avg_population, label = L"Average population, $\bar{p}$", linewidth=3, color=RGB(1,113/255,206/255),legend=:topright,ylims=(-0.001,0.021))

# Sublinear plotting
p1 = heatmap(heat, title = L"Population at time $t=200$", xaxis=false, yaxis=false, colorbar_title=L"Population, $p_i$", c = cgrad([RGB(1,1,1),RGB(1,113/255,206/255)]), clim=(0.0,0.05001),aspect_ratio=:equal)
p2 = plot(collect(0:T), lineSW, title= L"Utility $U_i$ vs time $t$", label=false, xlabel=L"Time, $t$", ylabel=L"Utility, $U_i$", color = :black, alpha=0.3)
plot!(collect(0:T),city_social_welfare, label = L"Average city utility, $W_C$", linewidth=3, color=RGB(1,113/255,206/255))
plot!(collect(0:T),utilitarian_social_welfare, label = L"Utilitarian, $W_U$", linewidth=3, color=RGB(1/255,205/255,254/255))
plot!(collect(0:T),Nash_social_welfare, label = L"Nash, $W_N$", linewidth=3, color=RGB(5/255,1,161/255), linestyle=:dash)
plot!(collect(0:T),Rawlsian_social_welfare, label = L"Rawlsian, $W_R$", linewidth=3, color=RGB(185/255,103/255,1),legend=:topleft,ylims=(-0.015,0.315))
p3 = plot(collect(0:T), linePop, title= L"Population $p_i$ vs time $t$", label=false, xlabel=L"Time, $t$", ylabel=L"Population, $p_i$", color = :black, alpha=0.3)
plot!(collect(0:T),avg_population, label = L"Average population, $\bar{p}$", linewidth=3, color=RGB(1,113/255,206/255),legend=:topleft,ylims=(-0.002,0.042))

plot(p1, p2, p3, layout=grid(1, 3), size=(1000,330))
# savefig("superlinear_Nash.pdf")
savefig("sublinear_Nash_single_budget.pdf")