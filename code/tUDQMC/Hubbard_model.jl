# using SU(2) ±1,±2 HS transformation
struct _Hubbard_Para
    Lattice::String
    t::Float64
    U::Float64
    site::Vector{Int64}
    Θ::Float64
    Ns::Int64
    Nt::Int64
    K::Array{Float64,2}
    BatchSize::Int64
    WrapTime::Int64
    Δt::Float64
    α::Float64
    γ::Vector{Float64}
    η::Vector{Float64}
    Pt::Array{Float64,2}
    HalfeK::Array{Float64,2}
    eK::Array{Float64,2}
    HalfeKinv::Array{Float64,2}
    eKinv::Array{Float64,2}
    nodes::Vector{Int64}
end

function Hubbard_Para(t, U, Lattice::String, site, Δt, Θ, BatchSize, Initial::String)
    Nt = 2 * cld(Θ, Δt)
    WrapTime = div(BatchSize, 2)
    
    α = sqrt(Δt * U / 2)
    γ = [1 + sqrt(6) / 3, 1 + sqrt(6) / 3, 1 - sqrt(6) / 3, 1 - sqrt(6) / 3]
    η = [sqrt(2 * (3 - sqrt(6))), -sqrt(2 * (3 - sqrt(6))), sqrt(2 * (3 + sqrt(6))), -sqrt(2 * (3 + sqrt(6)))]
    
    K = K_Matrix(Lattice, site)
    Ns = size(K, 1)

    E, V = LAPACK.syevd!('V', 'L',K[:,:])
    
    exp_neg_half = exp.(-Δt .* E ./ 2)
    exp_neg = exp.(-Δt .* E)
    exp_pos_half = exp.(Δt .* E ./ 2)
    exp_pos = exp.(Δt .* E)

    HalfeK = V * diagm(exp_neg_half) * V'
    eK = V * diagm(exp_neg) * V'
    HalfeKinv = V * diagm(exp_pos_half) * V'
    eKinv = V * diagm(exp_pos) * V'

    Pt = zeros(Float64, Ns, div(Ns, 2))  # 预分配 Pt
    if Initial == "H0"
        KK = copy(K)
        μ = 1e-5
        if occursin("HoneyComb", Lattice)
            KK .+= μ * diagm(repeat([-1, 1], div(Ns, 2)))
        elseif Lattice == "SQUARE"
            for i in 1:Ns
                x, y = i_xy(Lattice, site, i)
                KK[i, i] += μ * (-1)^(x + y)
            end
        end
        E, V = LAPACK.syevd!('V', 'L',KK)
        Pt .= V[:, 1:div(Ns, 2)]
    elseif Initial=="V" 
        if occursin("HoneyComb", Lattice)
            for i in 1:div(Ns,2)
                Pt[i*2,i]=1
            end
        else
            count=1
            for i in 1:Ns
                x,y=i_xy(Lattice,site,i)
                if (x+y)%2==1
                    Pt[i,count]=1
                    count+=1
                end
            end
        end
    end
    
    if div(Nt, 2) % BatchSize == 0
        nodes = collect(0:BatchSize:Nt)
    else
        nodes = vcat(0, reverse(collect(div(Nt, 2) - BatchSize:-BatchSize:1)), collect(div(Nt, 2):BatchSize:Nt), Nt)
    end

    return _Hubbard_Para(Lattice, t, U, site, Θ, Ns, Nt, K, BatchSize, WrapTime, Δt, α, γ, η, Pt, HalfeK, eK, HalfeKinv, eKinv, nodes)
end

function setμ(model::_Hubbard_Para, μ)
    N_particle = Int(round(μ^2 / (4 * π) * model.Ns))
    E, V = eigen(model.K)
    model.Pt .= V[:, 1:N_particle]
end
