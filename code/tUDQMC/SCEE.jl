# attractive-U and repulsive-U get the same S_2


function ctrl_SCEEicr(path::String,model::_Hubbard_Para,indexA::Vector{Int64},indexB::Vector{Int64},Sweeps::Int64,λ::Float64,Nλ::Int64,ss::Vector{Matrix{UInt8}},record)
    global LOCK=ReentrantLock()
    Ns=model.Ns
    ns=div(Ns, 2)
    NN=length(model.nodes)
    tau = Vector{ComplexF64}(undef, ns)
    ipiv = Vector{LAPACK.BlasInt}(undef, ns)
    ipivA = Vector{LAPACK.BlasInt}(undef, length(indexA))
    ipivB = Vector{LAPACK.BlasInt}(undef, length(indexB))
    II=Diagonal(ones(ComplexF64,Ns))

    name = if model.Lattice=="SQUARE" "□" 
        elseif model.Lattice=="HoneyComb60" "HC" 
        elseif model.Lattice=="HoneyComb120" "HC120" 
        else error("Lattice: $(model.Lattice) is not allowed !") end    
    file="$(path)/SCEEicr$(name)_t$(model.t)U$(model.U)size$(model.site)Δt$(model.Δt)Θ$(model.Θ)N$(Nλ)BS$(model.BatchSize).csv"
    
    # atexit() do 
    #     if record
    #         lock(LOCK) do
    #             open(file, "a") do io
    #                 writedlm(io, O', ',')
    #             end
    #         end
    #     end
    #     # writedlm("$(path)ss/SS$(name)_t$(model.t)U$(model.U)size$(model.site)Δt$(model.Δt)Θ$(model.Θ)λ$(Int(round(Nλ*λ))).csv", [ss[1] ss[2]],",")
    # end
    
    Gt1= Matrix{ComplexF64}(undef ,Ns, Ns)
    Gt2= Matrix{ComplexF64}(undef ,Ns, Ns)
    G01= Matrix{ComplexF64}(undef ,Ns, Ns)
    G02= Matrix{ComplexF64}(undef ,Ns, Ns)
    Gt01= Matrix{ComplexF64}(undef ,Ns, Ns)
    Gt02= Matrix{ComplexF64}(undef ,Ns, Ns)
    G0t1= Matrix{ComplexF64}(undef ,Ns, Ns)
    G0t2= Matrix{ComplexF64}(undef ,Ns, Ns)
    gmInv_A=Matrix{ComplexF64}(undef ,length(indexA),length(indexA))
    gmInv_B=Matrix{ComplexF64}(undef ,length(indexB),length(indexB))
    detg_A=detg_B=0.0

    b_A= Matrix{ComplexF64}(undef ,1,length(indexA))
    tmp1A= Matrix{ComplexF64}(undef ,1,length(indexA))
    tmpA1= Matrix{ComplexF64}(undef ,length(indexA),1)
    a_A= Matrix{ComplexF64}(undef ,length(indexA),1)

    b_B= Matrix{ComplexF64}(undef ,1,length(indexB))
    tmp1B= Matrix{ComplexF64}(undef ,1,length(indexB))
    tmpB1= Matrix{ComplexF64}(undef ,length(indexB),1)
    a_B= Matrix{ComplexF64}(undef ,length(indexB),1)

    # 预分配临时数组
    tmpN = Vector{ComplexF64}(undef, Ns)
    tmpN_ = Vector{ComplexF64}(undef, Ns)
    tmpNN = Matrix{ComplexF64}(undef, Ns, Ns)
    tmpNN2 = Matrix{ComplexF64}(undef, Ns, Ns)
    tmpNn = Matrix{ComplexF64}(undef, Ns, ns)
    tmpnn = Matrix{ComplexF64}(undef, ns, ns)
    tmpnN = Matrix{ComplexF64}(undef, ns, Ns)
    tmp1N = Matrix{ComplexF64}(undef ,1, Ns)
    tmpAA = Matrix{ComplexF64}(undef ,length(indexA),length(indexA))
    tmpBB = Matrix{ComplexF64}(undef ,length(indexB),length(indexB))

    rng=MersenneTwister(Threads.threadid()+time_ns())
    elements=(1, 2, 3, 4)
    samplers_dict = Dict{UInt8, Random.Sampler}()
    for excluded in elements
        allowed = [i for i in elements if i != excluded]
        samplers_dict[excluded] = Random.Sampler(rng, allowed)
    end
    
    tmpO=0.0
    counter=0
    O=zeros(Float64,Sweeps+1)
    O[1]=λ

    BMs1=Array{ComplexF64}(undef,Ns,Ns,NN-1)  # Number_of_BM*Ns*Ns
    BMs2=Array{ComplexF64}(undef,Ns,Ns,NN-1)  # Number_of_BM*Ns*Ns
    BMsinv1=Array{ComplexF64}(undef,Ns,Ns,NN-1)  # Number_of_BM*Ns*Ns
    BMsinv2=Array{ComplexF64}(undef,Ns,Ns,NN-1)  # Number_of_BM*Ns*Ns

    for idx in 1:NN-1
        BM_F!(tmpN,tmpNN,view(BMs1,:, : , idx),model,ss[1],idx)
        BM_F!(tmpN,tmpNN,view(BMs2,:,:,idx),model,ss[2],idx)
        BMinv_F!(tmpN,tmpNN,view(BMsinv1,:,:,idx),model,ss[1],idx)
        BMinv_F!(tmpN,tmpNN,view(BMsinv2,:,:,idx),model,ss[2],idx)
    end

    BLMs1=Array{ComplexF64}(undef,ns,Ns,NN)
    BRMs1=Array{ComplexF64}(undef,Ns,ns,NN)
    transpose!(view(BLMs1,:,:,NN) , model.Pt)
    copyto!(view(BRMs1,:,:,1) , model.Pt)
    
    BLMs2=Array{ComplexF64}(undef,ns,Ns,NN)
    BRMs2=Array{ComplexF64}(undef,Ns,ns,NN)
    transpose!(view(BLMs2,:,:,NN) , model.Pt)
    copyto!(view(BRMs2,:,:,1) , model.Pt)

    for i in 1:NN-1
        mul!(tmpnN,view(BLMs1,:,:,NN-i+1),view(BMs1,:,:,NN-i))
        LAPACK.gerqf!(tmpnN, tau)
        LAPACK.orgrq!(tmpnN, tau, ns)
        copyto!(view(BLMs1,:,:,NN-i) , tmpnN)
        
        mul!(tmpNn, view(BMs1,:,:,i), view(BRMs1,:,:,i))
        LAPACK.geqrf!(tmpNn, tau)
        LAPACK.orgqr!(tmpNn, tau, ns)
        copyto!(view(BRMs1,:,:,i+1) , tmpNn)
        # ---------------------------------------------------------------
        mul!(tmpnN,view(BLMs2,:,:,NN-i+1),view(BMs2,:,:,NN-i))
        LAPACK.gerqf!(tmpnN, tau)
        LAPACK.orgrq!(tmpnN, tau, ns)
        copyto!(view(BLMs2,:,:,NN-i) , tmpnN)

        mul!(tmpNn, view(BMs2,:,:,i), view(BRMs2,:,:,i))
        LAPACK.geqrf!(tmpNn, tau)
        LAPACK.orgqr!(tmpNn, tau, ns)
        copyto!(view(BRMs2,:,:,i+1) , tmpNn)

    end
    Θidx=div(NN,2)+1

    G4!(II,tmpnn,tmpNn,tmpNN,tmpNN2,ipiv,Gt1,G01,Gt01,G0t1,model.nodes,1,BLMs1,BRMs1,BMs1,BMsinv1)
    G4!(II,tmpnn,tmpNn,tmpNN,tmpNN2,ipiv,Gt2,G02,Gt02,G0t2,model.nodes,1,BLMs2,BRMs2,BMs2,BMsinv2)
    GroverMatrix!(gmInv_A,view(G01,indexA,indexA),view(G02,indexA,indexA))
    detg_A=abs2(det(gmInv_A))
    LAPACK.getrf!(gmInv_A,ipivA)
    LAPACK.getri!(gmInv_A, ipivA)
    GroverMatrix!(gmInv_B,view(G01,indexB,indexB),view(G02,indexB,indexB))
    detg_B=abs2(det(gmInv_B))
    LAPACK.getrf!(gmInv_B,ipivB)
    LAPACK.getri!(gmInv_B, ipivB)
    idx=1

    for loop in 1:Sweeps
        # println("\n ====== Sweep $loop / $Sweeps ======")
        for lt in 1:model.Nt
            @inbounds @simd for iii in 1:Ns
                @fastmath tmpN[iii] = cis( model.α *model.η[ss[1][iii, lt]] ) 
                @fastmath tmpN_[iii] = cis( model.α *model.η[ss[2][iii, lt]] ) 
            end

            WrapKV!(tmpNN,model.eK,model.eKinv,tmpN,Gt01,"Forward", "L")
            WrapKV!(tmpNN,model.eK,model.eKinv,tmpN_,Gt02,"Forward", "L")
            WrapKV!(tmpNN,model.eK,model.eKinv,tmpN,Gt1,"Forward", "B")
            WrapKV!(tmpNN,model.eK,model.eKinv,tmpN_,Gt2,"Forward", "B")
            WrapKV!(tmpNN,model.eK,model.eKinv,tmpN,G0t1,"Forward", "R")
            WrapKV!(tmpNN,model.eK,model.eKinv,tmpN_,G0t2,"Forward", "R")

            #####################################################################
            # Gt1_,G01_,Gt01_,G0t1_=G4_old(model,ss[1],lt,div(model.Nt,2))
            # Gt2_,G02_,Gt02_,G0t2_=G4_old(model,ss[2],lt,div(model.Nt,2))
                
            # if norm(Gt1-Gt1_)+norm(Gt2-Gt2_)+norm(Gt01-Gt01_)+norm(Gt02-Gt02_)+norm(G0t1-G0t1_)+norm(G0t2-G0t2_)>1e-3
            #     println( norm(Gt1-Gt1_),'\n',norm(Gt2-Gt2_),'\n',norm(Gt01-Gt01_),'\n',norm(Gt02-Gt02_),'\n',norm(G0t1-G0t1_),'\n',norm(G0t2-G0t2_) )
            #     error("WrapTime=$lt ")
            # end
            #####################################################################


            for x in 1:Ns
                # ss[1]
                begin
                    sx=rand(rng,samplers_dict[ss[1][x,lt]])
                
                    @fastmath Δ=cis(model.α*(model.η[sx]-model.η[ss[1][x,lt]]))-1
                    @fastmath r=1+Δ*(1-Gt1[x,x])
                    @fastmath p=model.γ[sx]/model.γ[ss[1][x,lt]]*abs2(r)
                    r=Δ/r

                    Tau_A=get_abTau1!(tmpAA,tmp1A,a_A,b_A,indexA,x,r,G02,Gt01,G0t1,gmInv_A)
                    Tau_B=get_abTau1!(tmpBB,tmp1B,a_B,b_B,indexB,x,r,G02,Gt01,G0t1,gmInv_B)

                    @fastmath p*= abs2(Tau_A)^λ * abs2(Tau_B)^(1-λ)
                                    
                    if rand(rng)<p
                        GMupdate!(tmpA1,tmpAA,a_A,b_A,Tau_A,gmInv_A)
                        GMupdate!(tmpB1,tmpBB,a_B,b_B,Tau_B,gmInv_B)
                        G4update!(tmpNN,tmp1N,x,r,Gt1,G01,Gt01,G0t1)
                        detg_A*=abs2(Tau_A)
                        detg_B*=abs2(Tau_B)

                        ss[1][x,lt]=sx
                        #####################################################################
                        # print('-')
                        
                        # Gt1_,G01_,Gt01_,G0t1_=G4_old(model,ss[1],lt,div(model.Nt,2))
                        # GM_A_=GroverMatrix(G01_[indexA,indexA],G02[indexA,indexA])
                        # gmInv_A_=inv(GM_A_)
                        # GM_B_=GroverMatrix(G01_[indexB,indexB],G02[indexB,indexB])
                        # gmInv_B_=inv(GM_B_)
                        # detg_A_=abs2(det(GM_A_))
                        # detg_B_=abs2(det(GM_B_))
                        # # println((G01_-G01)./(G0t1[:,x:x]*Gt01[x:x,:]))
                        # # println(1/Tau_A)
                        # # println((GM_A_-gmInv_A)./(a_A*b_A*gmInv_A))

                        # if norm(Gt1-Gt1_)+norm(G01-G01_)+norm(Gt01-Gt01_)+norm(G0t1-G0t1_)+
                        #    norm(gmInv_A_-gmInv_A)+norm(gmInv_B-gmInv_B_)+abs(detg_A-detg_A_)+abs(detg_B-detg_B_)>1e-3
                        #     println('\n',norm(Gt1-Gt1_),'\n',norm(G01-G01_),'\n',norm(Gt01-Gt01_),'\n',norm(G0t1-G0t1_))
                        #     println(norm(gmInv_A_-gmInv_A),' ',norm(gmInv_B-gmInv_B_),"\n",abs(detg_A-detg_A_),' ',abs(detg_B-detg_B_))
                        #     error("$lt  $x:,,,asdasdasd")
                        # end
                        #####################################################################
                    end
                end

                # ss[2]
                begin
                    sx=rand(rng,samplers_dict[ss[2][x,lt]])
                
                    @fastmath Δ=cis(model.α*(model.η[sx]-model.η[ss[2][x,lt]]))-1
                    @fastmath r=1+Δ*(1-Gt2[x,x])
                    @fastmath p=model.γ[sx]/model.γ[ss[2][x,lt]]*abs2(r)
                    r=Δ/r
                    Tau_A=get_abTau2!(tmpAA,a_A,b_A,indexA,x,r,G01,Gt02,G0t2,gmInv_A)
                    Tau_B=get_abTau2!(tmpBB,a_B,b_B,indexB,x,r,G01,Gt02,G0t2,gmInv_B)

                    @fastmath p*= abs2(Tau_A)^λ * abs2(Tau_B)^(1-λ)
                                    
                    if rand(rng)<p
                        GMupdate!(tmpA1,tmpAA,a_A,b_A,Tau_A,gmInv_A)
                        GMupdate!(tmpB1,tmpBB,a_B,b_B,Tau_B,gmInv_B)
                        G4update!(tmpNN,tmp1N,x,r,Gt2,G02,Gt02,G0t2)
                        detg_A*=abs2(Tau_A)
                        detg_B*=abs2(Tau_B)
                        ss[2][x,lt]=sx
                        # #####################################################################
                        # print('*')
                        # Gt2_,G02_,Gt02_,G0t2_=G4_old(model,ss[2],lt,div(model.Nt,2))
                        # GM_A_=GroverMatrix(G01[indexA,indexA],G02_[indexA,indexA])
                        # gmInv_A_=inv(GM_A_)
                        # GM_B_=GroverMatrix(G01[indexB,indexB],G02_[indexB,indexB])
                        # gmInv_B_=inv(GM_B_)
                        # detg_A_=abs2(det(GM_A_))
                        # detg_B_=abs2(det(GM_B_))

                        # if norm(Gt2-Gt2_)+norm(G02-G02_)+norm(Gt02-Gt02_)+norm(G0t2-G0t2_)+
                        #    norm(gmInv_A_-gmInv_A)+norm(gmInv_B-gmInv_B_)+abs(detg_A-detg_A_)+abs(detg_B-detg_B_)>1e-3
                        #     println('\n',norm(Gt2-Gt2_),'\n',norm(G02-G02_),'\n',norm(Gt02-Gt02_),'\n',norm(G0t2-G0t2_))
                        #     println(norm(gmInv_A_-gmInv_A),' ',norm(gmInv_B-gmInv_B_),"\n",abs(detg_A-detg_A_),' ',abs(detg_B-detg_B_))
                        #     error("$lt  $x:,,,asdasdasd")
                        # end
                        #####################################################################
                    end
                end
            end

            ##------------------------------------------------------------------------
            tmpO+=(detg_A/detg_B)^(1/Nλ)
            counter+=1
            ##------------------------------------------------------------------------

            if  any(model.nodes .== lt) 
                idx+=1
                BM_F!(tmpN,tmpNN,view(BMs1,:,:,idx-1),model,ss[1],idx-1)
                BMinv_F!(tmpN,tmpNN,view(BMsinv1,:,:,idx-1),model,ss[1],idx-1)
                BM_F!(tmpN,tmpNN,view(BMs2,:,:,idx-1),model,ss[2],idx-1)
                BMinv_F!(tmpN,tmpNN,view(BMsinv2,:,:,idx-1),model,ss[2],idx-1)
                for i in idx:max(Θidx,idx)
                    # println("update BR i=",i)
                    mul!(tmpNn, view(BMs1,:,:,i-1), view(BRMs1,:,:,i-1))
                    LAPACK.geqrf!(tmpNn,tau)
                    LAPACK.orgqr!(tmpNn, tau, ns)
                    copyto!(view(BRMs1,:,:,i) , tmpNn)
                    # ---------------------------------------------------------------
                    mul!(tmpNn, view(BMs2,:,:,i-1), view(BRMs2,:,:,i-1))
                    LAPACK.geqrf!(tmpNn,tau)
                    LAPACK.orgqr!(tmpNn, tau, ns)
                    copyto!(view(BRMs2,:,:,i) , tmpNn)
                end

                for i in idx-1:-1:min(Θidx,idx)
                    # println("update BL i=",i)
                    mul!(tmpnN,view(BLMs1,:,:,i+1),view(BMs1,:,:,i))
                    LAPACK.gerqf!(tmpnN,tau)
                    LAPACK.orgrq!(tmpnN, tau, ns)
                    copyto!(view(BLMs1,:,:,i) , tmpnN)
                    # ---------------------------------------------------------------
                    mul!(tmpnN,view(BLMs2,:,:,i+1),view(BMs2,:,:,i))
                    LAPACK.gerqf!(tmpnN,tau)
                    LAPACK.orgrq!(tmpnN, tau, ns)
                    copyto!(view(BLMs2,:,:,i) , tmpnN)
                end
                G4!(II,tmpnn,tmpNn,tmpNN,tmpNN2,ipiv,Gt1,G01,Gt01,G0t1,model.nodes,idx,BLMs1,BRMs1,BMs1,BMsinv1,"Forward")
                G4!(II,tmpnn,tmpNn,tmpNN,tmpNN2,ipiv,Gt2,G02,Gt02,G0t2,model.nodes,idx,BLMs2,BRMs2,BMs2,BMsinv2,"Forward")
                GroverMatrix!(gmInv_A,view(G01,indexA,indexA),view(G02,indexA,indexA))
                detg_A=abs2(det(gmInv_A))
                LAPACK.getrf!(gmInv_A,ipivA)
                LAPACK.getri!(gmInv_A, ipivA)
                GroverMatrix!(gmInv_B,view(G01,indexB,indexB),view(G02,indexB,indexB))
                detg_B=abs2(det(gmInv_B))
                LAPACK.getrf!(gmInv_B,ipivB)
                LAPACK.getri!(gmInv_B, ipivB)
            end

        end

        # println("\n ----------------reverse update ----------------")

        for lt in model.Nt:-1:1
            
            #####################################################################
            # Gt1_,G01_,Gt01_,G0t1_=G4_old(model,ss[1],lt,div(model.Nt,2))
            # Gt2_,G02_,Gt02_,G0t2_=G4_old(model,ss[2],lt,div(model.Nt,2))
            # if norm(Gt1-Gt1_)+norm(Gt2-Gt2_)+norm(Gt01-Gt01_)+norm(Gt02-Gt02_)+norm(G0t1-G0t1_)+norm(G0t2-G0t2_)>1e-3
            #     println( norm(Gt1-Gt1_),'\n',norm(Gt2-Gt2_),'\n',norm(Gt01-Gt01_),'\n',norm(Gt02-Gt02_),'\n',norm(G0t1-G0t1_),'\n',norm(G0t2-G0t2_) )
            #     error("WrapTime=$lt ")
            # end
            #####################################################################
            
            for x in 1:Ns
                # ss[1]
                begin
                    sx=rand(rng,samplers_dict[ss[1][x,lt]])
                
                    @fastmath Δ=cis(model.α*(model.η[sx]-model.η[ss[1][x,lt]]))-1
                    @fastmath r=1+Δ*(1-Gt1[x,x])
                    @fastmath p=model.γ[sx]/model.γ[ss[1][x,lt]]*abs2(r)
                    r=Δ/r

                    Tau_A=get_abTau1!(tmpAA,tmp1A,a_A,b_A,indexA,x,r,G02,Gt01,G0t1,gmInv_A)
                    Tau_B=get_abTau1!(tmpBB,tmp1B,a_B,b_B,indexB,x,r,G02,Gt01,G0t1,gmInv_B)

                    @fastmath p*= abs2(Tau_A)^λ * abs2(Tau_B)^(1-λ)
                                    
                    if rand(rng)<p
                        GMupdate!(tmpA1,tmpAA,a_A,b_A,Tau_A,gmInv_A)
                        GMupdate!(tmpB1,tmpBB,a_B,b_B,Tau_B,gmInv_B)
                        G4update!(tmpNN,tmp1N,x,r,Gt1,G01,Gt01,G0t1)
                        detg_A*=abs2(Tau_A)
                        detg_B*=abs2(Tau_B)

                        ss[1][x,lt]=sx
                        #####################################################################
                        # print('-')
                        
                        # Gt1_,G01_,Gt01_,G0t1_=G4_old(model,ss[1],lt,div(model.Nt,2))
                        # GM_A_=GroverMatrix(G01_[indexA,indexA],G02[indexA,indexA])
                        # gmInv_A_=inv(GM_A_)
                        # GM_B_=GroverMatrix(G01_[indexB,indexB],G02[indexB,indexB])
                        # gmInv_B_=inv(GM_B_)
                        # detg_A_=abs2(det(GM_A_))
                        # detg_B_=abs2(det(GM_B_))
                        # # println((G01_-G01)./(G0t1[:,x:x]*Gt01[x:x,:]))
                        # # println(1/Tau_A)
                        # # println((GM_A_-gmInv_A)./(a_A*b_A*gmInv_A))

                        # if norm(Gt1-Gt1_)+norm(G01-G01_)+norm(Gt01-Gt01_)+norm(G0t1-G0t1_)+
                        #    norm(gmInv_A_-gmInv_A)+norm(gmInv_B-gmInv_B_)+abs(detg_A-detg_A_)+abs(detg_B-detg_B_)>1e-3
                        #     println('\n',norm(Gt1-Gt1_),'\n',norm(G01-G01_),'\n',norm(Gt01-Gt01_),'\n',norm(G0t1-G0t1_))
                        #     println(norm(gmInv_A_-gmInv_A),' ',norm(gmInv_B-gmInv_B_),"\n",abs(detg_A-detg_A_),' ',abs(detg_B-detg_B_))
                        #     error("$lt  $x:,,,asdasdasd")
                        # end
                        #####################################################################
                    end
                end

                # ss[2]
                begin
                    sx=rand(rng,samplers_dict[ss[2][x,lt]])
                
                    @fastmath Δ=cis(model.α*(model.η[sx]-model.η[ss[2][x,lt]]))-1
                    @fastmath r=1+Δ*(1-Gt2[x,x])
                    @fastmath p=model.γ[sx]/model.γ[ss[2][x,lt]]*abs2(r)
                    r=Δ/r
                    Tau_A=get_abTau2!(tmpAA,a_A,b_A,indexA,x,r,G01,Gt02,G0t2,gmInv_A)
                    Tau_B=get_abTau2!(tmpBB,a_B,b_B,indexB,x,r,G01,Gt02,G0t2,gmInv_B)

                    @fastmath p*= abs2(Tau_A)^λ * abs2(Tau_B)^(1-λ)
                                    
                    if rand(rng)<p
                        GMupdate!(tmpA1,tmpAA,a_A,b_A,Tau_A,gmInv_A)
                        GMupdate!(tmpB1,tmpBB,a_B,b_B,Tau_B,gmInv_B)
                        G4update!(tmpNN,tmp1N,x,r,Gt2,G02,Gt02,G0t2)
                        detg_A*=abs2(Tau_A)
                        detg_B*=abs2(Tau_B)
                        ss[2][x,lt]=sx
                        # #####################################################################
                        # print('*')
                        # Gt2_,G02_,Gt02_,G0t2_=G4_old(model,ss[2],lt,div(model.Nt,2))
                        # GM_A_=GroverMatrix(G01[indexA,indexA],G02_[indexA,indexA])
                        # gmInv_A_=inv(GM_A_)
                        # GM_B_=GroverMatrix(G01[indexB,indexB],G02_[indexB,indexB])
                        # gmInv_B_=inv(GM_B_)
                        # detg_A_=abs2(det(GM_A_))
                        # detg_B_=abs2(det(GM_B_))

                        # if norm(Gt2-Gt2_)+norm(G02-G02_)+norm(Gt02-Gt02_)+norm(G0t2-G0t2_)+
                        #    norm(gmInv_A_-gmInv_A)+norm(gmInv_B-gmInv_B_)+abs(detg_A-detg_A_)+abs(detg_B-detg_B_)>1e-3
                        #     println('\n',norm(Gt2-Gt2_),'\n',norm(G02-G02_),'\n',norm(Gt02-Gt02_),'\n',norm(G0t2-G0t2_))
                        #     println(norm(gmInv_A_-gmInv_A),' ',norm(gmInv_B-gmInv_B_),"\n",abs(detg_A-detg_A_),' ',abs(detg_B-detg_B_))
                        #     error("$lt  $x:,,,asdasdasd")
                        # end
                        #####################################################################
                    end
                end
            end

            ##------------------------------------------------------------------------
            tmpO+=(detg_A/detg_B)^(1/Nλ)
            counter+=1
            ##------------------------------------------------------------------------

            if  any(model.nodes.== (lt-1)) 
                idx-=1
                BM_F!(tmpN,tmpNN,view(BMs1,:,:,idx),model,ss[1],idx)
                BM_F!(tmpN,tmpNN,view(BMs2,:,:,idx),model,ss[2],idx)
                BMinv_F!(tmpN,tmpNN,view(BMsinv1,:,:,idx),model,ss[1],idx)
                BMinv_F!(tmpN,tmpNN,view(BMsinv2,:,:,idx),model,ss[2],idx)
                for i in idx:-1:min(Θidx,idx)
                    # println("update BL i=",i)
                    mul!(tmpnN,view(BLMs1,:,:,i+1),view(BMs1,:,:,i))
                    LAPACK.gerqf!(tmpnN,tau)
                    LAPACK.orgrq!(tmpnN, tau, ns)
                    copyto!(view(BLMs1,:,:,i) , tmpnN)

                    mul!(tmpnN,view(BLMs2,:,:,i+1),view(BMs2,:,:,i))
                    LAPACK.gerqf!(tmpnN,tau)
                    LAPACK.orgrq!(tmpnN, tau, ns)
                    copyto!(view(BLMs2,:,:,i) , tmpnN)
                end
                for i in idx+1:max(Θidx,idx)
                    # println("update BR i=",i)
                    mul!(tmpNn, view(BMs1,:,:,i-1), view(BRMs1,:,:,i-1))
                    LAPACK.geqrf!(tmpNn,tau)
                    LAPACK.orgqr!(tmpNn, tau, ns)
                    copyto!(view(BRMs1,:,:,i) , tmpNn)

                    mul!(tmpNn, view(BMs2,:,:,i-1), view(BRMs2,:,:,i-1))
                    LAPACK.geqrf!(tmpNn,tau)
                    LAPACK.orgqr!(tmpNn, tau, ns)
                    copyto!(view(BRMs2,:,:,i) , tmpNn)
                end
                G4!(II,tmpnn,tmpNn,tmpNN,tmpNN2,ipiv,Gt1,G01,Gt01,G0t1,model.nodes,idx,BLMs1,BRMs1,BMs1,BMsinv1,"Backward")
                G4!(II,tmpnn,tmpNn,tmpNN,tmpNN2,ipiv,Gt2,G02,Gt02,G0t2,model.nodes,idx,BLMs2,BRMs2,BMs2,BMsinv2,"Backward")
                GroverMatrix!(gmInv_A,view(G01,indexA,indexA),view(G02,indexA,indexA))
                detg_A=abs2(det(gmInv_A))
                LAPACK.getrf!(gmInv_A,ipivA)
                LAPACK.getri!(gmInv_A, ipivA)
                GroverMatrix!(gmInv_B,view(G01,indexB,indexB),view(G02,indexB,indexB))
                detg_B=abs2(det(gmInv_B))
                LAPACK.getrf!(gmInv_B,ipivB)
                LAPACK.getri!(gmInv_B, ipivB)
            else
                @inbounds @simd for iii in 1:Ns
                    @fastmath tmpN[iii] = cis(- model.α *model.η[ss[1][iii, lt]] ) 
                    @fastmath tmpN_[iii] = cis(- model.α *model.η[ss[2][iii, lt]] ) 
                end
    
                WrapKV!(tmpNN,model.eK,model.eKinv,tmpN,Gt01,"Backward", "L")
                WrapKV!(tmpNN,model.eK,model.eKinv,tmpN_,Gt02,"Backward", "L")
                WrapKV!(tmpNN,model.eK,model.eKinv,tmpN,Gt1,"Backward", "B")
                WrapKV!(tmpNN,model.eK,model.eKinv,tmpN_,Gt2,"Backward", "B")
                WrapKV!(tmpNN,model.eK,model.eKinv,tmpN,G0t1,"Backward", "R")
                WrapKV!(tmpNN,model.eK,model.eKinv,tmpN_,G0t2,"Backward", "R")
            end
            
        end

        O[loop+1]=tmpO/counter
        tmpO=0.0
        counter=0
    end

    if record
        lock(LOCK) do
            open(file, "a") do io
                writedlm(io, O', ',')
            end
        end
    end

    return ss
end


function get_abTau1!(tmpAA::Matrix{ComplexF64},tmp1A,a::Matrix{ComplexF64},b::Matrix{ComplexF64},index::Vector{Int64},x::Int64,r::ComplexF64,G0::Matrix{ComplexF64},Gt0::Matrix{ComplexF64},G0t::Matrix{ComplexF64},gmInv::Matrix{ComplexF64})
    copyto!(tmpAA, view(G0,index,index))
    lmul!(2.0, tmpAA)
    for i in diagind(tmpAA)
        tmpAA[i] -= 1
    end
    mul!(tmp1A, view(Gt0,x:x,index), tmpAA)
    mul!(b, tmp1A, gmInv)
    copyto!(a,view(G0t,index,x))
    lmul!(r,b)
    Tau=dotu(a, b)+1
    return Tau
end


function get_abTau2!(tmpAA::Matrix{ComplexF64},a::Matrix{ComplexF64},b::Matrix{ComplexF64},index::Vector{Int64},x::Int64,r::ComplexF64,G0::Matrix{ComplexF64},Gt0::Matrix{ComplexF64},G0t::Matrix{ComplexF64},gmInv::Matrix{ComplexF64})
    copyto!(tmpAA, view(G0,index,index))
    lmul!(2.0, tmpAA)
    for i in diagind(tmpAA)
        tmpAA[i] -= 1
    end
    mul!(a,tmpAA,view(G0t,index,x))
    mul!(b,view(Gt0,x:x,index),gmInv)
    lmul!(r,a)
    Tau=dotu(a,b)+1
    return Tau
end

function G4update!(tmpNN::Matrix{ComplexF64},tmp1N::Matrix{ComplexF64},x::Int64,r::ComplexF64,Gt::Matrix{ComplexF64},G0::Matrix{ComplexF64},Gt0::Matrix{ComplexF64},G0t::Matrix{ComplexF64})
    mul!(tmpNN, view(G0t,:,x),view(Gt0,x:x,:))
    axpy!(r, tmpNN, G0)
    mul!(tmpNN, view(Gt,:,x),view(Gt0,x:x,:))
    axpy!(r, tmpNN, Gt0)

    copyto!(tmp1N , view(Gt,x:x, :))
    lmul!(-1.0, tmp1N)
    tmp1N[1, x] += 1
    mul!(tmpNN, view(G0t,:,x),tmp1N)
    axpy!(-r, tmpNN, G0t)
    mul!(tmpNN, view(Gt,:,x),tmp1N)
    axpy!(-r, tmpNN, Gt)
end

function GMupdate!(tmpA1::Matrix{ComplexF64},tmpAA::Matrix{ComplexF64},a::Matrix{ComplexF64},b::Matrix{ComplexF64},Tau::ComplexF64,gmInv::Matrix{ComplexF64})
    mul!(tmpA1, gmInv,a )
    mul!(tmpAA, tmpA1,b)
    axpy!(-1/Tau, tmpAA, gmInv)
end