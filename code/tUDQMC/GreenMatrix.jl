# 2d Trotter Decomposition
function BM_F!(tmpN,tmpNN,BM,model::_Hubbard_Para, s::Array{UInt8, 2}, idx::Int64)
    """
    不包头包尾
    """
    @assert 0< idx <=length(model.nodes)

    fill!(BM,0)
    @inbounds for i in diagind(BM)
        BM[i] = 1
    end
    for lt in model.nodes[idx] + 1:model.nodes[idx + 1]
        @inbounds @simd for i in 1:model.Ns
            tmpN[i] =  cis( model.α *model.η[s[i, lt]])
        end
        mul!(tmpNN,model.eK, BM)
        mul!(BM,Diagonal(tmpN), tmpNN)
    end
end

function BMinv_F!(tmpN,tmpNN,BM,model::_Hubbard_Para, s::Array{UInt8, 2}, idx::Int64)
    """
    不包头包尾
    """
    @assert 0< idx <=length(model.nodes)

    fill!(BM,0)
    @inbounds for i in diagind(BM)
        BM[i] = 1
    end

    for lt in model.nodes[idx] + 1:model.nodes[idx + 1]
        @inbounds for i in 1:model.Ns
            tmpN[i] =  cis( -model.α *model.η[s[i, lt]])
        end
        mul!(tmpNN,BM, model.eKinv)
        mul!(BM,tmpNN,Diagonal(tmpN))
    end
end

function G4!(II,tmpnn,tmpNn,tmpNN,tmpNN2,ipiv,Gt::Matrix{ComplexF64},G0::Matrix{ComplexF64},Gt0::Matrix{ComplexF64},G0t::Matrix{ComplexF64},nodes::Vector{Int64},idx::Int64,BLMs::Array{ComplexF64,3},BRMs::Array{ComplexF64,3},BMs::Array{ComplexF64,3},BMinvs::Array{ComplexF64,3},direction="Forward")
    Θidx=div(length(nodes),2)+1

    get_G!(tmpnn,tmpNn,ipiv,view(BLMs,:,:,idx),view(BRMs,:,:,idx),Gt)
    
    if idx==Θidx
        G0 .= Gt
        if direction=="Forward"
            Gt0.= Gt
            G0t.= Gt .- II 
        elseif direction=="Backward"
            Gt0.= Gt .- II 
            G0t.= Gt
        end
    else
        get_G!(tmpnn,tmpNn,ipiv,view(BLMs,:,:,Θidx),view(BRMs,:,:,Θidx),G0)
    
        Gt0 .= II
        G0t .= II
        if idx<Θidx
            for j in idx:Θidx-1
                if j==idx
                    tmpNN2 .= Gt
                else
                    get_G!(tmpnn,tmpNn,ipiv,view(BLMs,:,:,j),view(BRMs,:,:,j),tmpNN2)
                end
                mul!(tmpNN,tmpNN2, G0t)
                mul!(G0t, view(BMs,:,:,j), tmpNN)
                tmpNN .= II .- tmpNN2
                mul!(tmpNN2,Gt0, tmpNN)
                mul!(Gt0, tmpNN2, view(BMinvs,:,:,j))
                
            end
            lmul!(-1.0, Gt0)
        else
            for j in Θidx:idx-1
                if j==Θidx
                    tmpNN2 .= G0
                else
                    get_G!(tmpnn,tmpNn,ipiv,view(BLMs,:,:,j),view(BRMs,:,:,j),tmpNN2)
                end
                mul!(tmpNN, tmpNN2, Gt0)
                mul!(Gt0, view(BMs,:,:,j), tmpNN)
                tmpNN .= II .- tmpNN2
                mul!(tmpNN2, G0t, tmpNN)
                mul!(G0t, tmpNN2,view(BMinvs,:,:,j))
            end
            lmul!(-1.0, G0t)
        end        
    end
end

function GroverMatrix!(GM::Matrix{ComplexF64},G1::SubArray{ComplexF64, 2, Matrix{ComplexF64}, Tuple{Vector{Int64}, Vector{Int64}}, false},G2::SubArray{ComplexF64, 2, Matrix{ComplexF64}, Tuple{Vector{Int64}, Vector{Int64}}, false})
    mul!(GM,G1,G2)
    lmul!(2.0, GM)
    axpy!(-1.0, G1, GM)
    axpy!(-1.0, G2, GM)
    for i in diagind(GM)
        GM[i] += 1.0
    end
end

function Initial_s(model::_Hubbard_Para,rng::MersenneTwister)::Array{UInt8,2}
    sp=Random.Sampler(rng,[1,2,3,4])

    s::Array{UInt8,2}=zeros(model.Ns,model.Nt)

    for i = 1:model.Ns
        for j = 1:model.Nt
            # 从elements中随机选择一个元素来填充当前位置  
            s[i, j] =rand(rng,sp)
        end  
    end  

    return s
end

function Free_G!(G,Lattice,site,Θ,Initial)
    """
    input:
        Lattice: "HoneyComb" or "SQUARE"
        site: [Int64,Int64]
        Θ: ComplexF64
        Initial: "H0" or "V"
    return Green function with Free Hubbard Hamiltonian from H0 initial state or SDW initial state 
    对于得到H0初态(平衡态结果)
    必须要对H0加一个极其微弱的交错化学势,以去除基态简并,从而得到正确的结果
    """
    K=K_Matrix(Lattice,site)
    Ns=size(K)[1]
    ns=div(Ns, 2)

    Δt=0.01
    Nt=Int(round(Θ/Δt))

    Pt = zeros(ComplexF64, Ns, div(Ns, 2))  # 预分配 Pt
    if Initial=="H0"
        KK=K[:,:]
        μ=1e-5
        if occursin("HoneyComb", Lattice)
            KK+=μ*Diagonal(repeat([-1, 1], div(Ns, 2)))
        elseif Lattice=="SQUARE"
            for i in 1:Ns
                x,y=i_xy(Lattice,site,i)
                KK[i,i]+=μ*(-1)^(x+y)
            end
        end
        E,V=LAPACK.syevd!('V', 'L',KK[:,:])
        Pt=V[:,1:div(Ns,2)]
    elseif Initial=="V" 
        if occursin("HoneyComb", Lattice)
            for i in 1:Int(Ns/2)
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

    E,V=LAPACK.syevd!('V', 'L',K[:,:])
    eK=V*Diagonal(exp.(-Δt.*E))*V'

    BL = Array{ComplexF64}(undef, ns, Ns)
    BR = Array{ComplexF64}(undef, Ns, ns)
    tmpNn = Matrix{ComplexF64}(undef, Ns, ns)
    tmpnN = Matrix{ComplexF64}(undef, ns, Ns)
    tmpnn= Matrix{ComplexF64}(undef, ns, ns)
    tau = Vector{ComplexF64}(undef, ns)
    ipiv = Vector{LAPACK.BlasInt}(undef, ns)

    BL.=Pt'
    BR.=Pt

    count=0
    for i in 1:Nt
        mul!(tmpnN,BL,eK)
        BL.=tmpnN

        mul!(tmpNn,eK,BR)
        BR.=tmpNn

        count+=1
        if count ==10
            LAPACK.gerqf!(BL, tau)
            LAPACK.orgrq!(BL, tau, ns)

            LAPACK.geqrf!(BR, tau)
            LAPACK.orgqr!(BR, tau, ns)
            count=0
        end

    end

    mul!(tmpnn,BL,BR)
    LAPACK.getrf!(tmpnn,ipiv)
    LAPACK.getri!(tmpnn, ipiv)
    mul!(tmpNn,BR,tmpnn)
    mul!(G, tmpNn,BL)
    lmul!(-1.0,G)
    for i in diagind(G)
        G[i]+=1
    end
end


#######################################################################################################################################

function BM_F(model::_Hubbard_Para, s::Array{UInt8, 2}, idx::Int64)::Matrix{ComplexF64}
    """
    不包头包尾
    """
    Ns=model.Ns
    nodes=model.nodes
    η=model.η
    α=model.α

    @assert 0< idx <=length(model.nodes)

    fill!(tmpNN,0)
    @inbounds for i in diagind(tmpNN)
        tmpNN[i] = 1
    end
    for lt in nodes[idx] + 1:nodes[idx + 1]
        @inbounds begin
            for i in 1:Ns
                tmpN[i] =  cis( α *η[s[i, lt]])
            end
            mul!(tmpNN2,model.eK, tmpNN)
            mul!(tmpNN,Diagonal(tmpN), tmpNN2)
            # BM = Diagonal(D) * eK * BM
        end
    end

    return tmpNN
end

function BMinv_F(model::_Hubbard_Para, s::Array{UInt8, 2}, idx::Int64)::Matrix{ComplexF64}
    """
    不包头包尾
    """
    Ns=model.Ns
    nodes=model.nodes
    η=model.η
    α=model.α

    @assert 0< idx <=length(model.nodes)
    
    fill!(tmpNN,0)
    @inbounds for i in diagind(tmpNN)
        tmpNN[i] = 1
    end

    for lt in nodes[idx] + 1:nodes[idx + 1]
        @inbounds begin
            for i in 1:Ns
                tmpN[i] =  cis( -α *η[s[i, lt]])
            end
            mul!(tmpNN2,tmpNN, model.eKinv)
            mul!(tmpNN,tmpNN2,Diagonal(tmpN))
        end
    end

    return tmpNN
end

function G4(nodes,idx,BLMs,BRMs,BMs,BMinvs)
    Ns=size(BMs,1)
    Gt=Matrix{ComplexF64}(undef, Ns, Ns)
    Gt0=Matrix{ComplexF64}(undef, Ns, Ns)
    G0=Matrix{ComplexF64}(undef, Ns, Ns)
    G0t=Matrix{ComplexF64}(undef, Ns, Ns)

    Θidx=div(length(nodes),2)+1

    mul!(tmpnn,view(BLMs,:,:,idx), view(BRMs,:,:,idx))
    LAPACK.getrf!(tmpnn,ipiv)
    LAPACK.getri!(tmpnn, ipiv)
    mul!(tmpNn,view(BRMs,:,:,idx), tmpnn)
    mul!(tmpNN, tmpNn, view(BLMs,:,:,idx))
    Gt .= II .- tmpNN
    # Gt=II-BRMs[:,:,idx] * inv( BLMs[:,:,idx] * BRMs[:,:,idx] ) * BLMs[:,:,idx]
    
    mul!(tmpnn,view(BLMs,:,:,Θidx), view(BRMs,:,:,Θidx))
    LAPACK.getrf!(tmpnn,ipiv)
    LAPACK.getri!(tmpnn, ipiv)
    mul!(tmpNn,view(BRMs,:,:,Θidx), tmpnn)
    mul!(tmpNN, tmpNn, view(BLMs,:,:,Θidx))
    G0 .= II .- tmpNN
    # G0=II-BRMs[:,:,Θidx] * inv( BLMs[:,:,Θidx] * BRMs[:,:,Θidx] ) * BLMs[:,:,Θidx]
    
    Gt0 .= II
    G0t .= II
    if idx<Θidx
        for j in idx:Θidx-1
            if j==idx
                tmpNN .= II .- Gt
            else
                mul!(tmpnn,view(BLMs,:,:,j), view(BRMs,:,:,j))
                LAPACK.getrf!(tmpnn,ipiv)
                LAPACK.getri!(tmpnn, ipiv)
                mul!(tmpNn,view(BRMs,:,:,j), tmpnn)
                mul!(tmpNN, tmpNn, view(BLMs,:,:,j))
                # tmpNN=BRMs[:,:,j] *inv(BLMs[:,:,j] * BRMs[:,:,j])*BLMs[:,:,j]
            end
            mul!(tmpNN2,Gt0, tmpNN)
            mul!(Gt0, tmpNN2, view(BMinvs,:,:,j))
            # Gt0= Gt0* tmpNN*BMinvs[:,:,j]
            tmpNN2 .= II .- tmpNN
            mul!(tmpNN,tmpNN2, G0t)
            mul!(G0t, view(BMs,:,:,j), tmpNN)
            # G0t= BMs[:,:,j]*(II-tmpNN) * G0t
        end
        lmul!(-1.0, Gt0)
    elseif idx>Θidx
        for j in Θidx:idx-1
            if j==Θidx
                tmpNN .=II .- G0
            else
                mul!(tmpnn,view(BLMs,:,:,j), view(BRMs,:,:,j))
                LAPACK.getrf!(tmpnn,ipiv)
                LAPACK.getri!(tmpnn, ipiv)
                mul!(tmpNn,view(BRMs,:,:,j), tmpnn)
                mul!(tmpNN, tmpNn, view(BLMs,:,:,j))
                # tmp=BRMs[:,:,j]*inv(BLMs[:,:,j] * BRMs[:,:,j])* BLMs[:,:,j]
            end
            mul!(tmpNN2, G0t, tmpNN)
            mul!(G0t, tmpNN2,view(BMinvs,:,:,j))
            tmpNN2 .= II .- tmpNN
            mul!(tmpNN, tmpNN2, Gt0)
            mul!(Gt0, view(BMs,:,:,j), tmpNN)
            # G0t= G0t* tmp*BMinvs[j,:,:]
            # Gt0= BMs[j,:,:]*(II-tmp) * Gt0 
        end


        # for j in idx:-1:Θidx+1
            # if j==idx
            #     tmpNN .= II .- Gt
            # else
            #     mul!(tmpnn,view(BLMs,:,:,j), view(BRMs,:,:,j))
            #     LAPACK.getrf!(tmpnn,ipiv)
            #     LAPACK.getri!(tmpnn, ipiv)
            #     mul!(tmpNn,view(BRMs,:,:,j), tmpnn)
            #     mul!(tmpNN, tmpNn, view(BLMs,:,:,j))
            #     # tmp=BRMs[:,:,j]*inv(BLMs[:,:,j] * BRMs[:,:,j])* BLMs[:,:,j]
            # end
            # mul!(tmpNN2, tmpNN,G0t)
            # mul!(G0t, view(BMinvs,:,:,j-1),tmpNN2)
            # tmpNN2 .= II .- tmpNN
            # mul!(tmpNN,Gt0, view(BMs,:,:,j-1))
            # mul!(Gt0, tmpNN,tmpNN2)
            # # G0t= G0t* tmp*BMinvs[:,:,j]
            # # Gt0= BMs[:,:,j]*(II-tmp) * Gt0 
        lmul!(-1.0, G0t)
    else
        G0.=Gt
        Gt0.=Gt.-II
        G0t.=Gt
    end
    return Gt,G0,Gt0,G0t
end


function Gτ(nodes,lt,BLMs,BRMs)
    II=I(size(BLMs,3))
    idx=findfirst(nodes .== lt)
    if isnothing(idx)
        error("lt not in nodes")
    end
    G=II-BRMs[idx,:,:] * inv( BLMs[idx,:,:] * BRMs[idx,:,:] ) * BLMs[idx,:,:]
    return G
end




function Gτ_old(model::_Hubbard_Para,s::Array{UInt8,2},τ::Int64)::Array{ComplexF64,2}
    """
    equal time Green function
    """
    BL::Array{ComplexF64,2}=model.Pt'[:,:]
    BR::Array{ComplexF64,2}=model.Pt[:,:]

    counter=0
    for i in model.Nt:-1:τ+1
        D=[model.η[x] for x in s[:,i]]
        BL=BL*diagm(exp.(1im*model.α.*D))*model.eK
        counter+=1
        if counter==model.BatchSize
            counter=0
            BL=Matrix(qr(BL').Q)'
        end
    end
    counter=0
    for i in 1:1:τ
        D=[model.η[x] for x in s[:,i]]
        BR=diagm(exp.(1im*model.α.*D))*model.eK*BR
        counter+=1
        if counter==model.BatchSize
            counter=0
            BR=Matrix(qr(BR).Q)
        end
    end

    BL=Matrix(qr(BL').Q)'
    BR=Matrix(qr(BR).Q)

    return I(model.Ns)-BR*inv(BL*BR)*BL
end


function G4_old(model::_Hubbard_Para,s::Array{UInt8,2},τ1::Int64,τ2::Int64)
    """
    displaced Green function
    return:
        G(τ₁),G(τ₂),G(τ₁,τ₂),G(τ₂,τ₁)
    """
    if τ1>τ2
        BBs=zeros(ComplexF64,cld(τ1-τ2,model.BatchSize),model.Ns,model.Ns)
        BBsInv=zeros(ComplexF64,size(BBs))
        
        UL=zeros(ComplexF64,1+size(BBs)[1],div(model.Ns,2),model.Ns)
        UR=zeros(ComplexF64,size(UL)[1],model.Ns,div(model.Ns,2))
        G=zeros(ComplexF64,size(UL)[1],model.Ns,model.Ns)

        UL[end,:,:]=model.Pt'[:,:]
        UR[1,:,:]=model.Pt[:,:]
    
        counter=0
        for i in 1:τ2
            D=[model.η[x] for x in s[:,i]]
            UR[1,:,:]=diagm(exp.(1im*model.α.*D))*model.eK*UR[1,:,:]
            counter+=1
            if counter==model.BatchSize
                counter=0
                UR[1,:,:]=Matrix(qr(UR[1,:,:]).Q)
            end
        end
        # UR[1,:,:]=UR[1,:,:]
        UR[1,:,:]=Matrix(qr(UR[1,:,:]).Q)
    
        counter=0
        for i in model.Nt:-1:τ1+1
            D=[model.η[x] for x in s[:,i]]
            UL[end,:,:]=UL[end,:,:]*diagm(exp.(1im*model.α.*D))*model.eK
            counter+=1
            if counter==model.BatchSize
                counter=0
                UL[end,:,:]=Matrix(qr(UL[end,:,:]').Q)'
            end
        end
        # UL[end,:,:]=UL[end,:,:]
        UL[end,:,:]=Matrix(qr(UL[end,:,:]').Q)'
    
        for i in 1:size(BBs)[1]-1
            BBs[i,:,:]=I(model.Ns)
            BBsInv[i,:,:]=I(model.Ns)
            for j in 1:model.BatchSize
                D=[model.η[x] for x in s[:,τ2+(i-1)*model.BatchSize+j]]
                BBs[i,:,:]=diagm(exp.(1im*model.α.*D))*model.eK*BBs[i,:,:]
                BBsInv[i,:,:]=BBsInv[i,:,:]*model.eKinv*diagm(exp.(-1im*model.α.*D))
            end
        end
    
        BBs[end,:,:]=I(model.Ns)
        BBsInv[end,:,:]=I(model.Ns)
        for j in τ2+(size(BBs)[1]-1)*model.BatchSize+1:τ1
            D=[model.η[x] for x in s[:,j]]
            BBs[end,:,:]=diagm(exp.(1im*model.α.*D))*model.eK*BBs[end,:,:]
            BBsInv[end,:,:]=BBsInv[end,:,:]*model.eKinv*diagm(exp.(-1im*model.α.*D))
        end
    
        for i in 1:size(BBs)[1]
            UL[end-i,:,:]=Matrix(qr( (UL[end-i+1,:,:]*BBs[end-i+1,:,:])' ).Q)'
            UR[i+1,:,:]=Matrix(qr(BBs[i,:,:]*UR[i,:,:]).Q)
        end
    
        for i in 1:size(G)[1]
            G[i,:,:]=I(model.Ns)-UR[i,:,:]*inv(UL[i,:,:]*UR[i,:,:])*UL[i,:,:]
            #####################################################################
            # if i <size(G)[1]
            #     if norm(Gτ(model,s,τ2+(i-1)*model.BatchSize)-G[i,:,:])>1e-3
            #         error("$i Gt")
            #     end
            # else
            #     if norm(Gτ(model,s,τ1)-G[i,:,:])>1e-3
            #         error("$i Gt")
            #     end
            # end
            #####################################################################
        end


        G12=I(model.Ns)
        G21=-I(model.Ns)
        for i in 1:size(BBs)[1]
            G12=G12*BBs[end-i+1,:,:]*G[end-i,:,:]
            G21=G21*( I(model.Ns)-G[i,:,:] )*BBsInv[i,:,:]
        end
        
        return G[end,:,:],G[1,:,:],G12,G21
    
    elseif τ1<τ2
        G2,G1,G21,G12=G4_old(model,s,τ2,τ1)
        return G1,G2,G12,G21
    else
        G=Gτ_old(model,s,τ1)
        return G,G,-(I(model.Ns)-G),G
    
    end

end



function GroverMatrix(G1,G2)
    n = size(G1, 1)
    GM=Matrix{ComplexF64}(undef,n,n)
    
    mul!(GM,G1,G2)
    lmul!(2.0, GM)
    axpy!(-1.0, G1, GM)
    axpy!(-1.0, G2, GM)
    for i in diagind(GM)
        GM[i] += 1.0
    end
    return GM   
    # 2*G1*G2 - G1 - G2 + II
end




# function G12FF(model,s,τ1,τ2)
#     """
#     Debug G(τ1,τ2) Green function
#     Without numerical stablility
#     Only for short time debug!!!
#     """
#     if τ1>τ2
#         G=Gτ_old(model,s,τ2)
#         BBs=I(model.Ns)
#         BBsInv=I(model.Ns)

#         for i in τ2+1:τ1
#             D=[model.η[x] for x in s[:,i]]
#             BBs=diagm(exp.(1im*model.α.*D))*model.eK*BBs
#             BBsInv=BBsInv*model.eKinv*diagm(exp.(-1im*model.α.*D))
#         end


#         G12=BBs*G
#         G21=-( I(model.Ns)-G ) * BBsInv

#         return G12,G21
#     elseif τ1<τ2
#         G21,G12=G12FF(model,s,τ2,τ1)
#         return G12,G21
#     end
    
# end

