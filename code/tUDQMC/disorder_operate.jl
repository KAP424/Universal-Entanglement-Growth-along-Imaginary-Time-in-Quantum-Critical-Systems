
function SC_DOP(path::String,model::_Hubbard_Para,ω,indexA::Vector{Int64},indexB::Vector{Int64},Sweeps::Int64,λ::Float64,Nλ::Int64,s::Matrix{UInt8},record)::Matrix{UInt8}
    if model.Lattice=="SQUARE"
        name="□"
    elseif model.Lattice=="HoneyComb60"
        name="HC60"
    elseif model.Lattice=="HoneyComb120"
        name="HC120"
    end
    file="$(path)SCDOP$(name)_t$(model.t)U$(model.U)size$(model.site)Δt$(model.Δt)Θ$(model.Θ)N$(Nλ)BS$(model.BatchSize)ω$(round(ω,digits=2)).csv"
    atexit() do
        if record
            open(file, "a") do io
                lock(io)
                writedlm(io, O', ',')
                unlock(io)
            end
        end
        writedlm("$(path)s/S$(name)_t$(model.t)U$(model.U)size$(model.site)Δt$(model.Δt)Θ$(model.Θ)λ$(Int(round(Nλ*λ)))ω$(round(ω,digits=2)).csv", s,",")
    end

    rng=MersenneTwister(Threads.threadid()+round(Int,time()*1000))
    elements=(1, 2, 3, 4)

    Gt=zeros(ComplexF64,model.Ns,model.Ns)
    G0=zeros(ComplexF64,model.Ns,model.Ns)
    Gt0=zeros(ComplexF64,model.Ns,model.Ns)
    G0t=zeros(ComplexF64,model.Ns,model.Ns)
    XAinv=zeros(ComplexF64,length(indexA),length(indexA))
    XBinv=zeros(ComplexF64,length(indexB),length(indexB))
    detXA=0
    detXB=0

    tmpO=0
    counter=0
    O=zeros(Sweeps+1)
    O[1]=λ

    II=I(model.Ns)
    IA=I(length(indexA))
    IB=I(length(indexB))

    for loop in 1:Sweeps
        for lt in 1:model.Nt
            if mod(lt,model.WrapTime)==1 || lt==div(model.Nt,2)+1
                Gt,G0,Gt0,G0t=G4(model,s,lt,div(model.Nt,2))
                XAinv=inv( G0[indexA[:],indexA[:]]+exp(1im*ω)*(IA-G0[indexA[:],indexA[:]]) )
                XBinv=inv( G0[indexB[:],indexB[:]]+exp(1im*ω)*(IB-G0[indexB[:],indexB[:]]) )
                detXA=1/abs2(det(XAinv))
                detXB=1/abs2(det(XBinv))
            else
                D=[model.η[x] for x in s[:,lt]]
                Gt=diagm(exp.(1im*model.α.*D))*model.eK *Gt* model.eKinv*diagm(exp.(-1im*model.α.*D))
                G0t=G0t*model.eKinv*diagm(exp.(-1im*model.α.*D))
                Gt0=diagm(exp.(1im*model.α.*D))*model.eK*Gt0

                #####################################################################
                # Gt_,G0_,Gt0_,G0t_=G4(model,s,lt,div(model.Nt,2))
                    
                # if norm(Gt-Gt_)+norm(Gt0-Gt0_)+norm(G0t-G0t_)>1e-3
                #     println( norm(Gt-Gt_),'\n',norm(Gt0-Gt0_),'\n',norm(G0t-G0t_),'\n')
                #     error("$lt : WrapTime")
                # end
                #####################################################################
            end

            for x in 1:model.Ns
                b_A=transpose(Gt0[x,indexA[:]]) *XAinv
                a_A=G0t[indexA[:],x]
                Tau_A=b_A*a_A

                b_B=transpose(Gt0[x,indexB[:]]) *XBinv
                a_B=G0t[indexB[:],x]
                Tau_B=b_B*a_B

                sp=Random.Sampler(rng,[i for i in elements if i != s[x,lt]])
                sx=rand(rng,sp)
                
                Δ=exp(1im*model.α*(model.η[sx]-model.η[s[x,lt]]))-1
                r=1+Δ*(1-Gt[x,x])

                c_A=1+(1-exp(1im*ω))*Δ*Tau_A
                c_B=1+(1-exp(1im*ω))*Δ*Tau_B
                p=model.γ[sx]/model.γ[s[x,lt]]*abs2(c_A)^λ*abs2(c_B)^(1-λ)

                if rand(rng)<p
                    rho_A=(1-exp(1im*ω))*Δ/r/c_A
                    XAinv-=rho_A* ( XAinv*a_A .* b_A)
                    detXA*=abs2(c_A)

                    rho_B=(1-exp(1im*ω))*Δ/r/c_B
                    XBinv-=rho_B* ( XBinv*a_B .* b_B)
                    detXB*=abs2(c_B)

                    G0+=Δ/r* (G0t[:,x] .* transpose(Gt0[x,:]))
                    Gt0+=Δ/r* (Gt[:,x] .* transpose(Gt0[x,:]))
                    G0t-=Δ/r* (G0t[:,x] .* transpose( (II-Gt)[x,:] ) )
                    Gt-=Δ/r* (Gt[:,x] .* transpose( (II-Gt)[x,:]) )         
                    s[x,lt]=sx

                    #####################################################################
                    # print('-')
                    # Gt_,G0_,Gt0_,G0t_=G4(model,s,lt,div(model.Nt,2))
                    
                    # if norm(Gt-Gt_)+norm(G0-G0_)+norm(Gt0-Gt0_)+norm(G0t-G0t_)>1e-3
                    #     println('\n',norm(Gt-Gt_),'\n',norm(G0-G0_),'\n',norm(Gt0-Gt0_),'\n',norm(G0t-G0t_))
                    #     error("$lt  $x:,,,asdasdasd")
                    # end
                    #####################################################################
                end

            end

            ##------------------------------------------------------------------------
            tmpO+=(detXA/detXB)^(1/Nλ)
            counter+=1
            ##------------------------------------------------------------------------
        end

        for lt in model.Nt-1:-1:1
            if mod(lt,model.WrapTime)==0 || lt==div(model.Nt,2)
                Gt,G0,Gt0,G0t=G4(model,s,lt,div(model.Nt,2))
                XAinv=inv( G0[indexA[:],indexA[:]]+exp(1im*ω)*(IA-G0[indexA[:],indexA[:]]) )
                XBinv=inv( G0[indexB[:],indexB[:]]+exp(1im*ω)*(IB-G0[indexB[:],indexB[:]]) )
                detXA=1/abs2(det(XAinv))
                detXB=1/abs2(det(XBinv))
            else
                D=[model.η[x] for x in s[:,lt+1]]
                Gt=model.eKinv*diagm(exp.(-1im*model.α.*D)) *Gt* diagm(exp.(1im*model.α.*D))*model.eK
                G0t=G0t*diagm(exp.(1im*model.α.*D))*model.eK 
                Gt0=model.eKinv*diagm(exp.(-1im*model.α.*D)) *Gt0
                #####################################################################
                # Gt_,G0_,Gt0_,G0t_=G4(model,s,lt,div(model.Nt,2))
                    
                # if norm(Gt-Gt_)+norm(Gt0-Gt0_)+norm(G0t-G0t_)>1e-3
                #     println( norm(Gt-Gt_),'\n',norm(Gt0-Gt0_),'\n',norm(G0t-G0t_),'\n')
                #     error("$lt : WrapTime")
                # end
                #####################################################################
            end

            for x in 1:model.Ns
                b_A=transpose(Gt0[x,indexA[:]]) *XAinv
                a_A=G0t[indexA[:],x]
                Tau_A=b_A*a_A

                b_B=transpose(Gt0[x,indexB[:]]) *XBinv
                a_B=G0t[indexB[:],x]
                Tau_B=b_B*a_B

                sp=Random.Sampler(rng,[i for i in elements if i != s[x,lt]])
                sx=rand(rng,sp)
                
                Δ=exp(1im*model.α*(model.η[sx]-model.η[s[x,lt]]))-1
                r=1+Δ*(1-Gt[x,x])

                c_A=1+(1-exp(1im*ω))*Δ*Tau_A
                c_B=1+(1-exp(1im*ω))*Δ*Tau_B
                p=model.γ[sx]/model.γ[s[x,lt]]*abs2(c_A)^λ*abs2(c_B)^(1-λ)

                if rand(rng)<p
                    rho_A=(1-exp(1im*ω))*Δ/r/c_A
                    XAinv-=rho_A* ( XAinv*a_A .* b_A)
                    detXA*=abs2(c_A)

                    rho_B=(1-exp(1im*ω))*Δ/r/c_B
                    XBinv-=rho_B* ( XBinv*a_B .* b_B)
                    detXB*=abs2(c_B)

                    G0+=Δ/r* (G0t[:,x] .* transpose(Gt0[x,:]))
                    Gt0+=Δ/r* (Gt[:,x] .* transpose(Gt0[x,:]))
                    G0t-=Δ/r* (G0t[:,x] .* transpose( (II-Gt)[x,:] ) )
                    Gt-=Δ/r* (Gt[:,x] .* transpose( (II-Gt)[x,:]) )         
                    s[x,lt]=sx

                    #####################################################################
                    # print('-')
                    # Gt_,G0_,Gt0_,G0t_=G4(model,s,lt,div(model.Nt,2))
                    
                    # if norm(Gt-Gt_)+norm(G0-G0_)+norm(Gt0-Gt0_)+norm(G0t-G0t_)>1e-3
                    #     println('\n',norm(Gt-Gt_),'\n',norm(G0-G0_),'\n',norm(Gt0-Gt0_),'\n',norm(G0t-G0t_))
                    #     error("$lt  $x:,,,asdasdasd")
                    # end
                    #####################################################################
                end

            end
            tmpO+=(detXA/detXB)^(1/Nλ)
            counter+=1
        end
        O[loop+1]=tmpO/counter
        tmpO=counter=0
    end
    return s
end

function DOP_icr(path::String,model::_Hubbard_Para,ω,index::Vector{Int64},Sweeps::Int64,λ::Float64,Nλ::Int64,s::Matrix{UInt8},record)::Matrix{UInt8}
    if model.Lattice=="SQUARE"
        name="□"
    elseif model.Lattice=="HoneyComb60"
        name="HC60"
    elseif model.Lattice=="HoneyComb120"
        name="HC120"
    end
    file="$(path)DOP$(name)_t$(model.t)U$(model.U)size$(model.site)Δt$(model.Δt)Θ$(model.Θ)N$(Nλ)BS$(model.BatchSize)ω$(round(ω,digits=2))N$(length(index)).csv"
    
    atexit() do
        if record
            open(file, "a") do io
                lock(io)
                writedlm(io, O', ',')
                unlock(io)
            end
        end
        writedlm("$(path)s/S$(name)_t$(model.t)U$(model.U)size$(model.site)Δt$(model.Δt)Θ$(model.Θ)λ$(Int(round(Nλ*λ)))ω$(round(ω,digits=2)).csv", s,",")
    end
    
    rng=MersenneTwister(Threads.threadid()+round(Int,time()*1000))
    elements=(1, 2, 3, 4)

    Gt=zeros(ComplexF64,model.Ns,model.Ns)
    G0=zeros(ComplexF64,model.Ns,model.Ns)
    Gt0=zeros(ComplexF64,model.Ns,model.Ns)
    G0t=zeros(ComplexF64,model.Ns,model.Ns)
    Xinv=zeros(ComplexF64,length(index),length(index))
    detX=0

    tmpO=0
    counter=0
    O=zeros(Sweeps+1)
    O[1]=λ

    I1=I(model.Ns)
    I2=I(length(index))

    
    for loop in 1:Sweeps
        for lt in 1:model.Nt
            if mod(lt,model.WrapTime)==1 || lt==div(model.Nt,2)+1
                Gt,G0,Gt0,G0t=G4(model,s,lt,div(model.Nt,2))
                Xinv=inv( G0[index[:],index[:]]+exp(1im*ω)*(I2-G0[index[:],index[:]]) )
                detX=1/abs2(det(Xinv))
            else
                D=[model.η[x] for x in s[:,lt]]
                Gt=diagm(exp.(1im*model.α.*D))*model.eK *Gt* model.eKinv*diagm(exp.(-1im*model.α.*D))
                G0t=G0t*model.eKinv*diagm(exp.(-1im*model.α.*D))
                Gt0=diagm(exp.(1im*model.α.*D))*model.eK*Gt0

                #####################################################################
                # Gt_,G0_,Gt0_,G0t_=G4(model,s,lt,div(model.Nt,2))
                    
                # if norm(Gt-Gt_)+norm(Gt0-Gt0_)+norm(G0t-G0t_)>1e-3
                #     println( norm(Gt-Gt_),'\n',norm(Gt0-Gt0_),'\n',norm(G0t-G0t_),'\n')
                #     error("$lt : WrapTime")
                # end
                #####################################################################


            end

            for x in 1:model.Ns
                b=transpose(Gt0[x,index[:]]) *Xinv
                a=G0t[index[:],x]
                Tau=b*a

                sp=Random.Sampler(rng,[i for i in elements if i != s[x,lt]])
                sx=rand(rng,sp)
                
                Δ=exp(1im*model.α*(model.η[sx]-model.η[s[x,lt]]))-1
                r=1+Δ*(1-Gt[x,x])

                c=1+(1-exp(1im*ω))*Δ/r*Tau
                p=model.γ[sx]/model.γ[s[x,lt]]*abs2(r)*abs2(c)^(λ)

                if rand(rng)<p
                    rho=(1-exp(1im*ω))*Δ/r/c
                    Xinv-=rho* ( Xinv*a .* b)
                    detX*=abs2(c)

                    G0+=Δ/r* (G0t[:,x] .* transpose(Gt0[x,:]))
                    Gt0+=Δ/r* (Gt[:,x] .* transpose(Gt0[x,:]))
                    G0t-=Δ/r* (G0t[:,x] .* transpose( (I1-Gt)[x,:] ) )
                    Gt-=Δ/r* (Gt[:,x] .* transpose( (I1-Gt)[x,:]) )         
                    s[x,lt]=sx

                    #####################################################################
                    # print('-')
                    # Gt_,G0_,Gt0_,G0t_=G4(model,s,lt,div(model.Nt,2))
                    
                    # if norm(Gt-Gt_)+norm(G0-G0_)+norm(Gt0-Gt0_)+norm(G0t-G0t_)>1e-3
                    #     println('\n',norm(Gt-Gt_),'\n',norm(G0-G0_),'\n',norm(Gt0-Gt0_),'\n',norm(G0t-G0t_))
                    #     error("$lt  $x:,,,asdasdasd")
                    # end
                    #####################################################################
                end

            end

            ##------------------------------------------------------------------------
            tmpO+=detX^(1/Nλ)
            counter+=1
            ##------------------------------------------------------------------------
        end

        for lt in model.Nt-1:-1:1
            if mod(lt,model.WrapTime)==0 || lt==div(model.Nt,2)
                Gt,G0,Gt0,G0t=G4(model,s,lt,div(model.Nt,2))
                Xinv=inv( G0[index[:],index[:]]+exp(1im*ω)*(I2-G0[index[:],index[:]]) )
                detX=1/abs2(det(Xinv))
            else
                D=[model.η[x] for x in s[:,lt+1]]
                Gt=model.eKinv*diagm(exp.(-1im*model.α.*D)) *Gt* diagm(exp.(1im*model.α.*D))*model.eK
                G0t=G0t*diagm(exp.(1im*model.α.*D))*model.eK 
                Gt0=model.eKinv*diagm(exp.(-1im*model.α.*D)) *Gt0

                #####################################################################
                # Gt_,G0_,Gt0_,G0t_=G4(model,s,lt,div(model.Nt,2))
                    
                # if norm(Gt-Gt_)+norm(Gt0-Gt0_)+norm(G0t-G0t_)>1e-3
                #     println( norm(Gt-Gt_),'\n',norm(Gt0-Gt0_),'\n',norm(G0t-G0t_),'\n')
                #     error("$lt : WrapTime")
                # end
                #####################################################################
                end

                for x in 1:model.Ns
                    b=transpose(Gt0[x,index[:]]) *Xinv
                    a=G0t[index[:],x]
                    Tau=b*a
    
                    sp=Random.Sampler(rng,[i for i in elements if i != s[x,lt]])
                    sx=rand(rng,sp)
                    
                    Δ=exp(1im*model.α*(model.η[sx]-model.η[s[x,lt]]))-1
                    r=1+Δ*(1-Gt[x,x])
    
                    c=1+(1-exp(1im*ω))*Δ/r*Tau
                    p=model.γ[sx]/model.γ[s[x,lt]]*abs2(r)*abs2(c)^(λ)
    
                    if rand(rng)<p
                        rho=(1-exp(1im*ω))*Δ/r/c
                        Xinv-=rho* ( Xinv*a .* b)
                        detX*=abs2(c)
    
                        G0+=Δ/r* (G0t[:,x] .* transpose(Gt0[x,:]))
                        Gt0+=Δ/r* (Gt[:,x] .* transpose(Gt0[x,:]))
                        G0t-=Δ/r* (G0t[:,x] .* transpose( (I1-Gt)[x,:] ) )
                        Gt-=Δ/r* (Gt[:,x] .* transpose( (I1-Gt)[x,:]) )         
                        s[x,lt]=sx
    
                        #####################################################################
                        # print('-')
                        # Gt_,G0_,Gt0_,G0t_=G4(model,s,lt,div(model.Nt,2))
                        
                        # if norm(Gt-Gt_)+norm(G0-G0_)+norm(Gt0-Gt0_)+norm(G0t-G0t_)>1e-3
                        #     println('\n',norm(Gt-Gt_),'\n',norm(G0-G0_),'\n',norm(Gt0-Gt0_),'\n',norm(G0t-G0t_))
                        #     error("$lt  $x:,,,asdasdasd")
                        # end
                        #####################################################################
                    end
    
                end

            ##------------------------------------------------------------------------
            tmpO+=detX^(1/Nλ)
            counter+=1
            ##------------------------------------------------------------------------
        end

        O[loop+1]=tmpO/counter
        tmpO=counter=0

    end
    
    return s
end