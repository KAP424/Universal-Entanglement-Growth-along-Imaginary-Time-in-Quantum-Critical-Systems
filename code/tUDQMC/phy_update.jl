

function phy_update(path::String, model::_Hubbard_Para, WarmSweeps::Int64, Sweeps::Int64, s::Array{UInt8, 2})
    Ns=model.Ns
    ns=div(Ns, 2)
    NN=length(model.nodes)
    tau = Vector{ComplexF64}(undef, ns)
    ipiv = Vector{LAPACK.BlasInt}(undef, ns)


    name = if model.Lattice=="SQUARE" "□" 
    elseif model.Lattice=="HoneyComb60" "HC" 
    elseif model.Lattice=="HoneyComb120" "HC120" 
    else error("Lattice: $(model.Lattice) is not allowed !") end  

    rng = MersenneTwister(Threads.threadid()+time_ns())
    elements = (1, 2, 3, 4)
    samplers_dict = Dict{UInt8, Random.Sampler}()
    for excluded in elements
        allowed = [i for i in elements if i != excluded]
        samplers_dict[excluded] = Random.Sampler(rng, allowed)
    end

    mA = mB = nn = R0 = R1 = Ek = C0 = Cmax = 0.0
    counter = 0

    G = Matrix{ComplexF64}(undef ,Ns, Ns)

    # 预分配 BL 和 BR
    BLs = Array{ComplexF64}(undef, ns, Ns,NN)
    BRs = Array{ComplexF64}(undef, Ns, ns,NN)

    # 预分配临时数组
    tmpN = Vector{ComplexF64}(undef, Ns)
    tmpNN = Matrix{ComplexF64}(undef, Ns, Ns)
    BM = Matrix{ComplexF64}(undef, Ns, Ns)
    tmpNn = Matrix{ComplexF64}(undef, Ns, ns)
    tmpnn = Matrix{ComplexF64}(undef, ns, ns)
    tmpnN = Matrix{ComplexF64}(undef, ns, Ns)
    tmp1N = Matrix{ComplexF64}(undef, 1, Ns)

    copyto!(view(BRs,:,:,1) , model.Pt)
    transpose!(view(BLs,:,:,NN) , model.Pt)
    for idx in NN-1:-1:1
        BM_F!(tmpN,tmpNN,BM,model, s, idx)
        mul!(tmpnN,view(BLs,:,:,idx+1), BM)
        LAPACK.gerqf!(tmpnN, tau)
        LAPACK.orgrq!(tmpnN, tau, ns)
        copyto!(view(BLs,:,:,idx) , tmpnN)
        # view(BLs,:,:,idx) .= Matrix(qr!(tmpNn).Q)'
    end
    
    idx=1
    get_G!(tmpnn,tmpNn,ipiv,view(BLs,:,:,idx),view(BRs,:,:,idx),G)
    for loop in 1:(Sweeps + WarmSweeps)
        # println("\n Sweep: $loop ")
        for lt in 1:model.Nt
            #####################################################################
            # # println("lt=",lt-1)
            # if norm(G-Gτ_old(model,s,lt-1))>1e-6 
            #     error(lt-1,"Wrap error:  ",norm(G-Gτ_old(model,s,lt-1)))
            # end
            #####################################################################

            @inbounds @simd for iii in 1:Ns
                tmpN[iii] =@fastmath cis( model.α *model.η[s[iii,lt]] ) 
            end
            WrapKV!(tmpNN,model.eK,model.eKinv,tmpN,G,"Forward", "B")
            # G= Diagonal(tmp_D) * model.eK * G * model.eKinv * Diagonal(conj(tmp_D))

            @inbounds @simd for x in 1:Ns
                sx = rand(rng,  samplers_dict[s[x, lt]])
                @fastmath Δ = cis( model.α * (model.η[sx] - model.η[s[x, lt]])) - 1
                @fastmath r = 1 + Δ * (1 - G[x, x])

                if rand(rng) < @fastmath model.γ[sx] / model.γ[s[x, lt]] * abs2(r)
                    r=Δ / r
                    Gupdate!(tmpNN, tmp1N, x, r, G)
                    s[x, lt] = sx
                    ####################################################################
                    # if norm(G-Gτ_old(model,s,lt))>1e-6
                    #     error("asd")
                    # end
                    #####################################################################
                end
            end
            # ---------------------------------------------------------------------------------------------------------
            # record physical quantities
            # ---------------------------------------------------------------------------------------------------------
    
            if any(model.nodes .== lt )
                idx+=1
                BM_F!(tmpN,tmpNN,BM,model, s, idx - 1)
                mul!(tmpNn, BM, view(BRs,:,:,idx-1))
                LAPACK.geqrf!(tmpNn, tau)
                LAPACK.orgqr!(tmpNn, tau, ns)
                copyto!(view(BRs,:,:,idx) , tmpNn)

                copyto!(tmpNN , G)

                get_G!(tmpnn,tmpNn,ipiv,view(BLs,:,:,idx),view(BRs,:,:,idx),G)
                #####################################################################
                axpy!(-1.0, G, tmpNN)  
                if norm(tmpNN)>1e-8
                    println("Warning for Batchsize Wrap Error : $(norm(tmpNN))")
                end
                #####################################################################
            end
        end

        for lt in model.Nt:-1:1
            #####################################################################
            # print("-")
            # if norm(G-Gτ_old(model,s,lt))>1e-6 
            #     error(lt," Wrap error:  ",norm(G-Gτ_old(model,s,lt)))
            # end
            #####################################################################

            @inbounds @simd for x in 1:Ns
                sx = rand(rng,  samplers_dict[s[x, lt]])
                @fastmath Δ = cis( model.α * (model.η[sx] - model.η[s[x, lt]])) - 1
                @fastmath r = 1 + Δ * (1 - G[x, x])

                if rand(rng) < @fastmath model.γ[sx] / model.γ[s[x, lt]] * abs2(r)
                    r=Δ / r
                    Gupdate!(tmpNN, tmp1N, x, r, G)
                    s[x, lt] = sx
                    ####################################################################
                    # if norm(G-Gτ_old(model,s,lt))>1e-6
                    #     error("asd")
                    # end
                    #####################################################################
                end
            end
            # ---------------------------------------------------------------------------------------------------------
            # record physical quantities
            # ---------------------------------------------------------------------------------------------------------
            @inbounds @simd for iii in 1:Ns
                tmpN[iii] =@fastmath cis(-model.α *model.η[s[iii,lt]] ) 
            end
            WrapKV!(tmpNN,model.eK,model.eKinv,tmpN,G,"Backward", "B")
            
            if any(model.nodes.== (lt-1))
                # println("idx=",idx," lt=",lt-1)
                idx-=1
                BM_F!(tmpN,tmpNN,BM,model, s, idx)
                mul!(tmpnN,view(BLs,:,:,idx+1),BM)
                LAPACK.gerqf!(tmpnN, tau)
                LAPACK.orgrq!(tmpnN, tau, ns)
                copyto!(view(BLs,:,:,idx),tmpnN)

                get_G!(tmpnn,tmpNn,ipiv,view(BLs,:,:,idx),view(BRs,:,:,idx),G)
            end
        end

        if loop > WarmSweeps
            fid = open("$(path)/Phy$(name)_t$(model.t)U$(model.U)size$(model.site)Δt$(model.Δt)Θ$(model.Θ)BS$(model.BatchSize).csv", "a+")
            writedlm(fid, [Ek model.U * nn nn mA mB R0 R1 C0 Cmax] / counter, ',')
            close(fid)
            mA = mB = nn = R0 = R1 = Ek = C0 = Cmax = 0
            counter = 0
        end
    end
    return s
end

"""
    No Return. Overwrite G 
        G = I - BR ⋅ inv(BL ⋅ BR) ⋅ BL 
    ------------------------------------------------------------------------------
"""
function get_G!(tmpnn,tmpNn,ipiv,BL,BR,G)
    mul!(tmpnn, BL,BR)
    LAPACK.getrf!(tmpnn,ipiv)
    LAPACK.getri!(tmpnn, ipiv)
    mul!(tmpNn, BR, tmpnn)
    mul!(G, tmpNn, BL)
    lmul!(-1.0,G)
    for i in diagind(G)
        G[i]+=1
    end
end

function WrapKV!(tmpNN,eK,eKinv,D,G,direction,LR)
    if direction=="Forward"
        if LR=="L"
            mul!(tmpNN, eK, G)
            mul!(G,Diagonal(D),tmpNN)
        elseif LR=="R"
            mul!(tmpNN, G,eKinv)
            mul!(G,tmpNN , Diagonal(D))
        elseif LR=="B"
            mul!(tmpNN, eK, G)
            mul!(G,tmpNN,eKinv)
            mul!(tmpNN,Diagonal(D),G)
            conj!(D)
            mul!(G,tmpNN,Diagonal(D))
        end
    elseif direction=="Backward"
        if LR=="L"
            mul!(tmpNN,Diagonal(D),G)
            mul!(G, eKinv, tmpNN)
        elseif LR=="R"
            mul!(tmpNN,G,Diagonal(D))
            mul!(G, tmpNN,eK)
        elseif LR=="B"
            mul!(tmpNN,Diagonal(D),G)
            conj!(D)
            mul!(G,tmpNN,Diagonal(D))
            mul!(tmpNN, eKinv, G)
            mul!(G,tmpNN,eK)
        end
    end
end

function Gupdate!(tmpNN::Matrix{ComplexF64},tmp1N::Matrix{ComplexF64},x::Int64,r::ComplexF64,G::Matrix{ComplexF64})
    view(tmp1N,1, :) .= .-view(G,x, :)
    tmp1N[1, x] += 1
    mul!(tmpNN, view(G, :, x), tmp1N)
    axpy!(-r, tmpNN, G)
end


# function Poss(model,s)
#     A=model.Pt[:,:]

#     for i in 1:model.Nt
#         D=[model.η[x] for x in s[:,i]]
#         A=diagm(exp.(1im*model.α.*D))*model.eK*A
#     end
#     A=model.Pt'*A

#     ans=det(A)

#     return abs2(ans)
    
# end
