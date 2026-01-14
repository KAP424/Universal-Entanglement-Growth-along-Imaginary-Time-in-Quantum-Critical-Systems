push!(LOAD_PATH, "/home/zxli_1/KAP/SOURCE/tUDQMC")
using Distributed,Dates

cpus = parse(Int, ENV["SLURM_CPUS_PER_TASK"])
addprocs(cpus)
println(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),"  ",cpus, " processes added.")


@everywhere using LinearAlgebra.BLAS

@everywhere BLAS.set_num_threads(1)

@everywhere using KAPDQMC_tU
@everywhere using Random
@everywhere using Printf

function parse_simple_array_string(s::String)  
    # 移除字符串两端的方括号  
    s = s[2:end-1]  
      
    # 以逗号为分隔符分割字符串  
    elements = split(s, ",")  
      
    # 将每个元素转换为整数并收集到数组中  
    result = [parse(Int64, element) for element in elements]  
      
    return result  
end 


t=parse(Float64,ARGS[1]);   U=parse(Float64,ARGS[2]);   Lattice=ARGS[3];
site=parse_simple_array_string(ARGS[4]);   Δt=parse(Float64,ARGS[5]);    Θ=parse(Float64,ARGS[6]);   
BatchSize=parse(Int64,ARGS[7]);    Sweeps=parse(Int64,ARGS[8]);   N=parse(Int64,ARGS[9]);
lambda=parse(Float64,ARGS[10]);         path=ARGS[11];       Pt=ARGS[12];
    
TTT=time_ns()

model=Hubbard_Para(t,U,Lattice,site,Δt,Θ,BatchSize,Pt)

TTT=round(Int,(time_ns()-TTT)/1e9)
h = TTT ÷ 3600
m = (TTT % 3600) ÷ 60
s = TTT % 60
println("model finished in ",@sprintf("%02d:%02d:%02d", h, m, s))

@everywhere model = $model

L=site[1]
# Half
indexA=area_index(Lattice,site,([1,1],[div(L,3),L]))
@everywhere indexA = $indexA

indexB=area_index(Lattice,site,([1,1],[div(L,3),div(2*L,3)]))
@everywhere indexB = $indexB

T1=time_ns()
@sync begin
    @distributed for i in 1:cpus
        TTT=time_ns()
        ##############################################################
        rng = MersenneTwister(time_ns()+myid())
        s1=Initial_s(model,rng)
        s2=Initial_s(model,rng)
        global_ss=ctrl_SCEEicr(path,model,indexA,indexB,2,lambda,N,[s1[:,:],s2[:,:]],false)
        # global_ss=[s1[:,:],s2[:,:]]
        ##############################################################
        ctrl_SCEEicr(path,model,indexA,indexB,Sweeps,lambda,N,global_ss,true)

        TTT=round(Int,(time_ns()-TTT)/1e9)
        h = TTT ÷ 3600
        m = (TTT % 3600) ÷ 60
        s = TTT % 60
        println("size$(model.site)  Δt$(model.Δt)  Θ$(model.Θ)", "  Sweep $(Sweeps+2) finished in ",@sprintf("%02d:%02d:%02d", h, m, s))
    end
end

T2=round(Int,(time_ns()-T1)/1e9)
h = T2 ÷ 3600
m = (T2 % 3600) ÷ 60
s = T2 % 60
println(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),"  ","All processes finished in ",@sprintf("%02d:%02d:%02d", h, m, s),"\n")
exit()