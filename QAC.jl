using DWave#, Plots


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
    dir::String="test/", inst_list::Union{Array{Int64}, UnitRange{Int64}} = [], loadInsts::Bool = false, loadInstsFile::String = "instances.jld", verbose::Bool = false,#working info
    problem_size::Int64=288, num_insts::Int64=5, betas::Array{Float64}=[0.1, 0.2, 0.3, 0.4, 0.5], #QAC vs. C info
    annealing_times::Array{Int64}=[5], num_gauges::Int64=20, num_runs_DW::Int64=20, num_samples_DW::Int64=1000, num_runs_HFS::Int64=20,
    num_samples_HFS::Int64=1000, run_HFS::Bool=true, run_C::Bool=true, run_QAC::Bool=true, mark_annealing_time::Bool = true) #experiment info

    #Create the directory if it doesn't exist
    dir[end] == '/' || (dir = string(dir, '/'))
    mkpath(dir)

    
    isfile(string(solverName, "_code.jld")) || generatePudenzCode(solver=solverName, codeName=solverName)
    
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


    rep_emb = PudenzEmbedding(beta = 0.0, repCode = true, problemSize = problem_size, codeName = solverName)

    p_emb = []
    for b in betas
        push!(p_emb, PudenzEmbedding(beta = b, problemSize = problem_size, codeName = solverName))
    end
            
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

            #eg = EmbeddingGenerator([PudenzEmbedding=>GaugeEmbedding=>solver_size])
            for a = 1:length(annealing_times)

                props[:annealing_time] = annealing_times[a]

                fn = string("instC_", i)
                mark_annealing_time && (fn = string(fn, "_", annealing_times[a]))

                exper = ExperimentDescription(instance=applyEmbedding(rep_emb, inst), R=4*num_samples_DW, name=fn, solver=:DWave,
                    solver_properties=props,embedding_generator=eg, decode_opts=[1])
                run_experiment(exper, minruns=num_runs_DW,maxruns=num_runs_DW,vc=0,vcmode=:prob, N=num_gauges,
                        savesols=true,save_encoded=false,tosave=true, 
                        save_embeddings=false,coupleruns=false,savepath=dir,timeout=2.,verbose=verbose)
            end
        end



        # Pudenz Code
        #-------------
        if run_QAC
            for a = 1:length(annealing_times)

                props[:annealing_time] = annealing_times[a]
                for j = 1:length(betas)
                    verbose && println("Running Pudenz Code on DWave with beta = ", betas[j])

                    fn = string("instQAC_", i)
                    mark_annealing_time && (fn = string(fn, "_", annealing_times[a]))

                    exper = ExperimentDescription(instance=applyEmbedding(p_emb[j], inst), R=num_samples_DW, name=fn,solver=:DWave, 
                        solver_properties=props,embedding_generator=eg, decode_opts=[1])
                    run_experiment(exper, minruns=num_runs_DW,maxruns=num_runs_DW,vc=0,vcmode=:prob, N=num_gauges,
                            savesols=true,save_encoded=false,tosave=true, 
                            save_embeddings=false,coupleruns=false,savepath=dir,timeout=2.,verbose=verbose) 
                end
            end
        end
    end

    println("Done with ", inst_list, " at ",  Dates.format(now(), "HH:MM (m/d)"))
end


#Create a dataset of Pudenz instances of a given size
function createDataSet(; file_name::String = "instances", gen_code::Bool = false, solver_name::String = "DW2X", num_insts::Int64 = 1000, problem_size::Int64 = 288, total_size::Int64 = 288)

    instances = Array{ProblemInstance}(num_insts)

    for i = 1:num_insts
        instances[i] = generatePudenzInstance(total_size, numBitsUsed = problem_size, solverName = solver_name)
    end

    save(string(file_name, ".jld"), "instances", instances)
    nothing
end


#Unembed the solutions found by DWave
function unembed(;dir::String="", inst_list::Union{Array{Int64}, UnitRange{Int64}}=[1], problem_size::Int64 = 288, 
    betas::Array{Float64} = [0.1, 0.2, 0.3, 0.4, 0.5], codeName::String = "DW2X", annealing_times::Array{Int64} = [5],
    unembed_C::Bool = true, unembed_QAC::Bool = true, verbose::Bool = false, mark_annealing_time::Bool = true)


    dir[end] == '/' || (dir = string(dir, '/'))

    instances = load(string(dir, "instances.jld"),"instances")
    #The C/Rep Code unembedding is the same for all instances
    unembed_C && (cemb = PudenzEmbedding(beta = 0.0, repCode = true, codeName = codeName))

    if unembed_QAC
        pemb = []
        for b in betas
            push!(pemb, PudenzEmbedding(beta = b, problemSize = problem_size, codeName = codeName))
        end
    end

    for i in inst_list

        inst = instances[i]
        verbose && println(i)

        #Unembed C Instances
        if unembed_C
            for a = 1:length(annealing_times)
                fn = string(dir, "instC_", i)
                mark_annealing_time && (fn = string(fn, "_", annealing_times[a]))
                fn = string(fn, "/1.jld2")
                res = load(fn)

                #Checking to be sure this instance hasn't been unembedded
                if size(res["decoded_solutions"], 1) != length(inst.h)

                    verbose && println("Unembedding C for annealing time = ", annealing_times[a])

                    (numSamps, numRuns) = size(res["energies"])
                    newE = Array{Float64}(numSamps, numRuns)
                    newSols = BitArray(length(inst.h), numSamps, numRuns)
                    inst_tmp = applyEmbedding(cemb, inst)

                    for j = 1:numRuns                   
                        (newE[:, j], newSols[:, :, j]) = reverseEmbedding(cemb, inst_tmp, res["decoded_solutions"][:, :, j])
                    end

                    res["decoded_solutions"] = newSols
                    res["decoded_energies"] = newE
                    save(fn, res)
                elseif verbose
                    println("Already unembedded C for annealing_time = ", annealing_times[a])
                end
            end
        end


        #Unembed QAC Instances
        if unembed_QAC
            for a = 1:length(annealing_times)
                for b = 1:length(betas)

                    fn = string(dir, "instQAC_", i)
                    mark_annealing_time && (fn = string(fn, "_", annealing_times[a]))
                    fn = string(fn, "/", b, ".jld2")
                    res = load(fn)


                    #Checking to be sure this instance hasn't been unembedded
                    if size(res["decoded_solutions"], 1) != length(inst.h)

                        verbose && println("Unembedding QAC-", betas[b], " annealing time = ", annealing_times[a])

                        (numSamps, numRuns) = size(res["energies"])
                        newE = Array{Float64}(numSamps, numRuns)
                        newSols = BitArray(length(inst.h), numSamps, numRuns)
                        
                        inst_tmp = applyEmbedding(pemb[b], inst)

                        for j = 1:numRuns                       
                            (newE[:, j], newSols[:, :, j]) = reverseEmbedding(pemb[b], inst_tmp, res["decoded_solutions"][:, :, j])
                        end

                        res["decoded_solutions"] = newSols
                        res["decoded_energies"] = newE
                        save(fn, res)
                    elseif verbose
                        println("Already unembedded QAC-", betas[b], " for annealing = ", annealing_times[a])
                    end
                end
            end
        end
    end
end


#Calculate the success probabilities across the various runs/tests
function analysis(; dir::String = "", inst_list::Union{Array{Int64}, UnitRange{Int64}} = [1],
    betas::Array{Float64} = [0.1 0.2 0.3 0.4 0.5], annealing_times::Array{Int64} = [5], num_posterior_samples::Int64 = 1000,
    prob_threshold::Float64 = 0.02, best_beta::Bool = true, verbose::Bool = false, use_HFS::Bool = true, mark_annealing_time::Bool = true)

    num_insts = length(inst_list)
    successC = Array{Float64}(num_insts, length(annealing_times), 3)
    successQAC = (best_beta ? Array{Float64}(num_insts, length(annealing_times), 4) : Array{Float64}(num_insts, length(annealing_times), length(betas), 4))


    dir[end] == '/' || (dir = string(dir, '/'))

    for ix = 1:num_insts
        i = inst_list[ix]


        #Find the Ground State Energy/Solution
        if use_HFS
            if isfile(string(dir, "instHFS_", i, "/1.jld2"))
                gs_e = minimum(load(string(dir, "instHFS_", i, "/1.jld2"))["decoded_energies"])
                verbose && println(i, " gs_e = ", gs_e, " (HFS)")
                which_solver = 1
            else
                gs_e = 100000000.0
                println("Unable to find ground state energy for instance ", i, ". Will use minimum found across DWave runs")
            end
        else
            gs_e = 1000000.0
        end
            
        resC = []
        resQ = []
        for a = 1:length(annealing_times)

            fn = string(dir, "instC_", i)
            mark_annealing_time && (fn = string(fn, "_", annealing_times[a]))
            fn = string(fn, "/1.jld2")
            push!(resC, load(fn))

            #for finding the minimum across all runs/tests
            tmp = gs_e
            gs_e = minimum([gs_e; resC[a]["decoded_energies"]])
            (verbose && tmp != gs_e) && println(i, " gs_e = ", gs_e, " (C), annealing_time = ", annealing_times[a])

            for b = 1:length(betas)
                fn = string(dir, "instQAC_", i)
                mark_annealing_time && (fn = string(fn, "_", annealing_times[a]))
                fn = string(fn, "/", b, ".jld2")
                push!(resQ, load(fn))

                tmp = gs_e
                gs_e = minimum([gs_e; resQ[(a - 1)*length(betas) + b]["decoded_energies"]])

                (verbose && tmp != gs_e) && println(i, " gs_e = ", gs_e, " (QAC, beta = ", betas[b], "), annealing_time = ", annealing_times[a])
            end
        end


        #Calculate success probabilities given the ground state energy found above for each annealing time separately
        for a = 1:length(annealing_times)

            ps = get_posteriors(resC[a]["decoded_energies"],false,num_posterior_samples,true_Egs=gs_e)[:,end]
            successC[ix, a, :] = [mean(ps), std(ps), gs_e]


            tempVals = [0.0 0.0 0.0 0.0]
            for b = 1:length(betas)

                ps = get_posteriors(resQ[(a - 1)*length(betas) + b]["decoded_energies"],false,num_posterior_samples,true_Egs=gs_e)[:,end]
                
                if best_beta
                    (mean(ps) - tempVals[1] > prob_threshold) && (tempVals = [mean(ps) std(ps) gs_e betas[b]])
                else
                    successQAC[ix, a, b, :] = [mean(ps) std(ps) gs_e betas[b]]
                end
            end

            best_beta && (successQAC[ix, a, :] = tempVals)
        end

        verbose && println("\n\n")
    end

    save(string(dir, "successQAC.jld"), "success", successQAC)
    save(string(dir, "successC.jld"), "success", successC)
end


#=function plotResults( ; dir::String = "USC_288/", c_v_qac::Bool = true, betas::Array{Float64} = [0.1, 0.2, 0.3, 0.4, 0.5])

    dir[end] == '/' || (dir = string(dir, '/'))

    successC = load(string(dir, "successC.jld"))["success"]
    successQAC = load(string(dir, "successQAC.jld"))["success"]

    # determine if only the best was picked (length = 2) or if we have the results for all betas (length = 3)
    if length(size(successQAC)) == 3

        for ix in 1=size(successQAC)[1]

            tempVals = [0.0 0.0 0.0]
            for j = 1:length(betas)

                if successQAC[]
            end
        end
    else
        sQ = successQAC
    end

    scatter(successC[:, 1], successQAC[:, 1], m = ColorGradient(:rainbow), zcolor = successQAC[:, 4], colorbar = true, legend = false)
    xmin = minimum(successC[:, 1]) - 0.015
    xmax = maximum(successC[:, 1]) + 0.02
    plot!(linspace(xmin, xmax), linspace(xmin, xmax))

    savefig(string(dir, "results"))
end=#