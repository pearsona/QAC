using DWave, Plots


#Changes to implement:
# NOT DECODING/REVERSE EMBEDDING THE QAC INSTANCES!!! (can use fixInsts.jl afterwards... but should bake into code!)
# -store/retreive embedding code files within storage directory
# -create more plots
# -ability to change annealing time (just parameter to ExperimentDescription)
# -option to automatically run HFS in analysis if not found

#Variable Clarifications:
# this file | Josh's DWave.jl | DWave | what it is
# -num_samples_* | R | num_reads | how many samples are taken
# -num_gauges | N | - | number of gauges to be used in each runs
# -num_runs_* | maxruns | - | maximum number of submissions to solver
#       NOTE: each gauge takes up a run (e.g. 10 gauges and 20 runs leads to 2 runs of each gauge)


function runQAC(token::String ; solverName::String="DW2X", url::String="https://usci.qcc.isi.edu/sapi", #solver info
    dir::String="test/", inst_list = [], loadInsts::Bool = false, loadInstsFile::String = "instances.jld", verbose::Bool = false,#working info
    problem_size::Int64=288, num_insts::Int64=5, betas::Array{Float64}=[0.1, 0.2, 0.3, 0.4, 0.5], #QAC vs. C info
    num_gauges::Int64=20, num_runs_DW::Int64=20, num_samples_DW::Int64=1000, num_runs_HFS::Int64=20, 
    num_samples_HFS::Int64=1000, run_HFS::Bool=true, run_C::Bool=true, run_QAC::Bool=true) #experiment info

    #Create the directory if it doesn't exist
    dir[end] == '/' || (dir = string(dir, '/'))
    mkpath(dir)

    
    isfile(string(solverName, "_code.jld")) || generatePudenzCode(solver=solv, codeName=solverName)
    
    if isfile(string(solverName, "_props.jld"))
        props = load(string(solverName, "_props.jld"))["props"]
    else
        props = DWave.dwrm.RemoteConnection(url, token)[:get_solver](solverName)[:properties]
        save(string(solverName, "_props.jld"), "props", props)
    end

    props[:token] = token
    props[:url] = url
    props[:solver_name] = solverName

    solver_size = props["num_qubits"] #USC DW2X = 1152, NASA 2000q = 2048
    log_solver_size = ceil(Int, solver_size/4)

    if loadInsts 
        instances = load(loadInstsFile, "instances")
        num_insts = length(instances)
    end


    eg = EmbeddingGenerator([GaugeEmbedding=>solver_size])
    HFS_eg = EmbeddingGenerator([GaugeEmbedding=>log_solver_size]) #need to create a gauge of largest size possible to work with code


    length(inst_list) == 0 && (inst_list = 1:num_insts)


    println("Starting")
    for i in inst_list  #@sync @parallel ??
        println(i, " at ",  Dates.format(now(), "HH:MM (m/d)"))

        loadInsts ? (inst = instances[i]) : (inst = generatePudenzInstance(log_solver_size, numBitsUsed = problem_size, solverName = solverName))

        # HFS for finding GS Energy
        #---------------------------
        if run_HFS #&& !isfile()
            verbose && println("Running HFS")
            exper = ExperimentDescription(instance=inst, R=num_samples_HFS, name=string("instHFS_", i),solver=:HFS, solver_properties=props, 
                embedding_generator=HFS_eg, decode_opts=[1])
            run_experiment(exper, minruns=1,maxruns=num_runs_HFS,N=num_gauges, vc=0,vcmode=:prob,
                savesols=false,save_encoded=false,tosave=true,savepath=dir,save_embeddings=false,coupleruns=false,timeout=2.,verbose=verbose)
        end


        
        # C/Repetition Code
        #-------------------
        # beta = 0 means no penalty bit (so rep code) and true means only one phys bit used (so 1 copy), but will get 4X the samples
        if run_C
            verbose && println("Running Repetition code")
            rep_emb = PudenzEmbedding(beta = 0.0, repCode = true, problemSize = problem_size, codeName = solverName)
            inst_c = applyEmbedding(rep_emb, inst)
            exper = ExperimentDescription(instance=inst_c, R=4*num_samples_DW, name=string("instC_", i), solver=:DWave,
                solver_properties=props,embedding_generator=eg, decode_opts=[1])
            run_experiment(exper, minruns=num_runs_DW,maxruns=num_runs_DW,vc=0,vcmode=:prob, N=num_gauges,
                    savesols=true,save_encoded=false,tosave=true, 
                    save_embeddings=false,coupleruns=false,savepath=dir,timeout=2.,verbose=verbose)
        end



        # Pudenz Code
        #-------------
        if run_QAC
            for b in betas
                # Create the physical problem for the given beta
                p_emb = PudenzEmbedding(beta = b, codeName = solverName)
                inst_p = applyEmbedding(p_emb, inst)
                
                # Send pudenz/QAC problem off to DWave
                verbose && println("Running Pudenz Code on DWave with beta = ", b)
                exper = ExperimentDescription(instance=inst_p, R=num_samples_DW, name=string("instQAC_", i),solver=:DWave, 
                    solver_properties=props,embedding_generator=eg, decode_opts=[1])
                run_experiment(exper, minruns=num_runs_DW,maxruns=num_runs_DW,vc=0,vcmode=:prob, N=num_gauges,
                        savesols=true,save_encoded=false,tosave=true, 
                        save_embeddings=false,coupleruns=false,savepath=dir,timeout=2.,verbose=verbose) 
            end
        end
    end

    println("Done with ", inst_list, " at ",  Dates.format(now(), "HH:MM (m/d)"))
end


function createDataSet(; file_name::String = "instances", gen_code::Bool = false, solver_name::String = "DW2X", num_insts::Int64 = 1000, problem_size::Int64 = 288, total_size::Int64 = 288)

    instances = Array{ProblemInstance}(num_insts)

    for i = 1:num_insts
        instances[i] = generatePudenzInstance(total_size, numBitsUsed = problem_size, solverName = solver_name)
    end

    save(string(file_name, ".jld"), "instances", instances)
    nothing
end




function analysis(; dir::String = "", inst_list::Array{Int64} = [1], verbose::Bool=false,
    betas::Array{Float64} = [0.1 0.2 0.3 0.4 0.5], num_posterior_samples::Int64 = 1000)

    num_insts = length(inst_list)
    successC = Array{Float64}(num_insts, 3)
    successQAC = Array{Float64}(num_insts, 4)

    #keep track of how many times each solver worked
    #1: HFS, 2: C, 3-7: QAC w/ betas
    counts = [0.0 0.0 0.0 0.0 0.0 0.0 0.0]
    which_solver = 0

    dir[end] == '/' || (dir = string(dir, '/'))

    for ix = 1:num_insts
        try
            i = inst_list[ix]


            if isfile(string(dir, "instHFS_", i, "/1.jld2"))
                gs_e = minimum(load(string(dir, "instHFS_", i, "/1.jld2"))["decoded_energies"])
                verbose && println(i, " gs_e = ", gs_e, " (HFS)")
                which_solver = 1
            else
                gs_e = 100000000.0
                println("Unable to find ground state energy for instance ", i, ". Will use minimum found across DWave runs")
            end
                

            resC = load(string(dir, "instC_", i, "/1.jld2"))
            tmp = gs_e
            gs_e = minimum([gs_e; resC["decoded_energies"]])
            (verbose && tmp != gs_e) && println(i, " gs_e = ", gs_e, " (C)")

            if abs(tmp - gs_e) >= 1e-9
                which_solver = 2
            end

            resQ = []
            for j = 1:length(betas)
                push!(resQ, load(string(dir, "instQAC_", i, "/", j, ".jld2")))
                tmp = gs_e
                gs_e = minimum([gs_e; resQ[j]["decoded_energies"]])
                (verbose && tmp != gs_e) && println(i, " gs_e = ", gs_e, " (QAC, beta = ", betas[j], ")")
                
                if abs(tmp - gs_e) >= 1e-9
                    which_solver = 2 + j
                end
            end

            counts[which_solver] = counts[which_solver] + 1

            ps = get_posteriors(resC["decoded_energies"],false,num_posterior_samples,true_Egs=gs_e)[:,end]
            successC[ix, :] = [mean(ps), std(ps), gs_e]

            tempVals = [0.0 0.0 0.0]
            for j = 1:length(betas)

                ps = get_posteriors(resQ[j]["decoded_energies"],false,num_posterior_samples,true_Egs=gs_e)[:,end]
                if mean(ps) > tempVals[1]
                    tempVals = [mean(ps) std(ps) betas[j]]
                end
            end

            successQAC[ix, :] = [tempVals[1], tempVals[2], gs_e, tempVals[3]]

            println("\n\n")
        catch e
            println(e)
            println(ix)
        end
    end

    println("Counts that HFS, C, QAC- 0.1, 0.2, 0.3, 0.4, 0.5 got the gs: ", counts, " out of ", num_insts)

    save(string(dir, "successQAC.jld"), "success", successQAC)
    save(string(dir, "successC.jld"), "success", successC)

    # Results
    #---------
    scatter(successC[:, 1], successQAC[:, 1], m = ColorGradient(:rainbow), zcolor = successQAC[:, 4], colorbar = true, legend = false)
    xmin = minimum(successC[:, 1]) - 0.015
    xmax = maximum(successC[:, 1]) + 0.02
    plot!(linspace(xmin, xmax), linspace(xmin, xmax))

    savefig(string(dir, "results"))
end