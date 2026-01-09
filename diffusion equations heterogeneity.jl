using DifferentialEquations, LaTeXStrings, LinearAlgebra, Plots

# Parameters
L = 10
T = 4000
Y₀ = 1
N = 1
α₁ = [1, 0]
α₂ = [0, 1]
β = [0.7,1.3] # sublinear scaling

# Calculate utility
function calc_utility(u,p,l)
    utility = zeros(L^2)
    for i=1:L
        for j=1:L
            for k=1:2
                utility[L*(i-1)+j] += p[l][k]*log(1.0 + (u[L*(i-1)+j]+u[L*(i-1)+j+L^2])^(1-p[3][k])/(Y₀*N^p[3][k]))
            end
        end
    end
    return utility
end

# Replicator equations
function replicator!(du,u,p,t)
    utility₁ = calc_utility(u,p,1)
    utility₂ = calc_utility(u,p,2)
    for i=1:L
        for j=1:L
            diff_utility₁ = 0
            diff_utility₂ = 0
            for m=1:L
                for n=1:L
                    if m!=i || n!=j
                        diff_utility₁ += u[L*(m-1)+n]*(utility₁[L*(i-1)+j] - utility₁[L*(m-1)+n])/((i-m)^2 + (j-n)^2)
                        diff_utility₂ += u[L*(m-1)+n+L^2]*(utility₂[L*(i-1)+j] - utility₂[L*(m-1)+n])/((i-m)^2 + (j-n)^2)
                    end
                end
            end
            du[L*(i-1)+j] = u[L*(i-1)+j]*diff_utility₁
            du[L*(i-1)+j+L^2] = u[L*(i-1)+j+L^2]*diff_utility₂
        end
    end
end

# Initial conditions and parameters
u0 = rand(2*L^2)
u0 = u0/sum(u0)
p = (α₁,α₂,β)
tspan = (0.0,T)

# Solve differential equations
prob = ODEProblem(replicator!,u0,tspan,p)
# sol = solve(prob,abstol=1e-20,reltol=1e-20,saveat=1)
sol = solve(prob,abstol=1e-20,reltol=1e-20,saveat=1)

# Plot results
pyplot()

# Format data for pop 1
total_pop₁ = sum(u0[1:L^2])
population₁ = vcat(hcat(u0[1:L^2], zeros(L^2)),zeros(Int(T)*(L^2),2))
social_welfare₁ = vcat(hcat(calc_utility(u0,p,1), zeros(L^2)),zeros(Int(T)*(L^2),2))
global count = L^2
for j = 2:T+1
    for i = 1:L^2
        global count = count + 1
        social_welfare₁[count,:] = [calc_utility(sol[:,j],p,1)[i], j]
        population₁[count,:] = [sol[i,j], j]
    end
end
city_social_welfare₁ = transpose(sum(reshape(social_welfare₁[:,1],L^2,T+1),dims=1)/L^2)
utilitarian_social_welfare₁ = transpose(sum(reshape(social_welfare₁[:,1].*population₁[:,1]/total_pop₁,L^2,T+1),dims=1))
Nash_social_welfare₁ = prod.(eachcol(reshape(social_welfare₁[:,1],L^2,T+1).^(reshape(population₁[:,1]/total_pop₁,L^2,T+1))))
Rawlsian_social_welfare₁ = transpose(minimum(reshape(social_welfare₁[:,1],L^2,T+1),dims=1))
avg_population₁ = transpose(sum(sol[1:L^2,:],dims=1)/L^2)
heat₁ = reshape(sol[1:L^2,Int(T)]/total_pop₁,L,L)
lineSW₁ = transpose(reshape(social_welfare₁[:,1],L^2,T+1))
linePop₁ = transpose(reshape(population₁[:,1]/total_pop₁,L^2,T+1))

# Sublinear plotting
p1 = heatmap(heat₁, title = L"Population at time $t=4000$", xaxis=false, yaxis=false, colorbar_title=L"Population, $q_i^H$", c = cgrad([RGB(1,1,1),RGB(1,113/255,206/255)]), clim=(0.0,1.05),aspect_ratio=:equal)
p2 = plot(collect(0:T), lineSW₁, title= L"Utility $U_i^H$ vs time $t$", label=false, xlabel=L"Time, $t$", ylabel=L"Utility, $U_i^H$", color = :black, alpha=0.3)
plot!(collect(0:T),city_social_welfare₁, label = L"Average city utility, $W_C^H$", linewidth=3, color=RGB(1,113/255,206/255))
plot!(collect(0:T),utilitarian_social_welfare₁, label = L"Utilitarian, $W_U^H$", linewidth=3, color=RGB(1/255,205/255,254/255))
plot!(collect(0:T),Nash_social_welfare₁, label = L"Nash, $W_N^H$", linewidth=3, color=RGB(5/255,1,161/255), linestyle=:dash)
plot!(collect(0:T),Rawlsian_social_welfare₁, label = L"Rawlsian, $W_R^H$", linewidth=3, color=RGB(185/255,103/255,1),legend=:topleft,ylims=(-0.0625,1.25))
p3 = plot(collect(0:T), linePop₁, title= L"Population $q_i^H$ vs time $t$", label=false, xlabel=L"Time, $t$", ylabel=L"Population, $q_i^H$", color = :black, alpha=0.3)
plot!(collect(0:T),avg_population₁, label = L"Average population, $\bar{q}_i^H$", linewidth=3, color=RGB(1,113/255,206/255),legend=:right,ylims=(-0.05,1.05),xlims=(-200,4200))

# Format data for pop 2
total_pop₂ = sum(u0[L^2+1:2*L^2])
population₂ = vcat(hcat(u0[L^2+1:2*L^2], zeros(L^2)),zeros(Int(T)*(L^2),2))
social_welfare₂ = vcat(hcat(calc_utility(u0,p,2), zeros(L^2)),zeros(Int(T)*(L^2),2))
global count = L^2
for j = 2:T+1
    for i = 1:L^2
        global count = count + 1
        social_welfare₂[count,:] = [calc_utility(sol[:,j],p,2)[i], j]
        population₂[count,:] = [sol[i+L^2,j], j]
    end
end
city_social_welfare₂ = transpose(sum(reshape(social_welfare₂[:,1],L^2,T+1),dims=1)/L^2)
utilitarian_social_welfare₂ = transpose(sum(reshape(social_welfare₂[:,1].*population₂[:,1]/total_pop₂,L^2,T+1),dims=1))
Nash_social_welfare₂ = prod.(eachcol(reshape(social_welfare₂[:,1],L^2,T+1).^(reshape(population₂[:,1]/total_pop₂,L^2,T+1))))
Rawlsian_social_welfare₂ = transpose(minimum(reshape(social_welfare₂[:,1],L^2,T+1),dims=1))
avg_population₂ = transpose(sum(sol[L^2+1:2*L^2,:]/total_pop₂,dims=1)/L^2)
heat₂ = reshape(sol[L^2+1:2*L^2,Int(T)]/total_pop₂,L,L)
lineSW₂ = transpose(reshape(social_welfare₂[:,1],L^2,T+1))
linePop₂ = transpose(reshape(population₂[:,1]/total_pop₂,L^2,T+1))

# Superlinear plotting
p4 = heatmap(heat₂, title = L"Population at time $t=4000$", xaxis=false, yaxis=false, colorbar_title=L"Population, $q_i^L$", c = cgrad([RGB(1,1,1),RGB(1/255,205/255,254/255)]), clim=(0.0,0.02001),aspect_ratio=:equal)
p5 = plot(collect(0:T), lineSW₂, title= L"Utility $U_i^L$ vs time $t$", label=false, xlabel=L"Time, $t$", ylabel=L"Utility, $U_i^L$", color = :black, alpha=0.3)
plot!(collect(0:T),city_social_welfare₂, label = L"Average city utility, $W_C^L$", linewidth=3, color=RGB(1,113/255,206/255))
plot!(collect(0:T),utilitarian_social_welfare₂, label = L"Utilitarian, $W_U^L$", linewidth=3, color=RGB(1/255,205/255,254/255))
plot!(collect(0:T),Nash_social_welfare₂, label = L"Nash, $W_N^L$", linewidth=3, color=RGB(5/255,1,161/255), linestyle=:dash)
plot!(collect(0:T),Rawlsian_social_welfare₂, label = L"Rawlsian, $W_R^L$", linewidth=3, color=RGB(185/255,103/255,1),legend=:topleft,ylims=(0.61,4.79),yticks = ([1,2,3,4], ["1.0", "2.0", "3.0","4.0"]))
p6 = plot(collect(0:T), linePop₂, title= L"Population $q_i^L$ vs time $t$", label=false, xlabel=L"Time, $t$", ylabel=L"Population, $q_i^L$", color = :black, alpha=0.3)
plot!(collect(0:T),avg_population₂, label = L"Average population, $\bar{q}_i^L$", linewidth=3, color=RGB(1,113/255,206/255),legend=:topright,ylims=(-0.0011,0.0231),xlims=(-200,4200))

plot(p1, p2, p3, layout=grid(1, 3), size=(1060,330))
savefig("2_pop_sublinear.pdf")

plot(p4, p5, p6, layout=grid(1, 3), size=(1060,330))
savefig("2_pop_superlinear.pdf")