using DWave, Plots


#Changes to implement:
# -store/retreive embedding code files within storage directory
# -create more plots
# -ability to change annealing time (just parameter to ExperimentDescription)

function runQAC(token::String ; solverName::String="DW2X", url::String="https://usci.qcc.isi.edu/sapi", #solver info
    dir::String="test/", inst_list = [], loadInsts::Bool = false, loadInstsFile::String = "instances.jld", verbose::Bool = false,#working info
    problem_size::Int64=288, num_insts::Int64=5, betas::Array{Float64}=[0.1, 0.2, 0.3, 0.4, 0.5], #QAC vs. C info
    num_per_call::Int64=10, num_runs_dw::Int64=20, num_samples_dw::Int64=1000, num_runs_hfs::Int64=20, 
    num_samples_hfs::Int64=1000, num_posterior_samples::Int64=1000, run_hfs::Bool=true, run_C::Bool=true, run_QAC::Bool=true) #experiment info

    #Create the directory if it doesn't exist
    dir[end] == '/' || (dir = string(dir, '/'))
    mkpath(dir)

    solv = DWave.dwrm.RemoteConnection(url, token)[:get_solver](solverName)
    isfile(string(solverName, "_code.jld")) || generatePudenzCode(solver=solv, codeName=solverName)
    props = solv[:properties]
    props[:token] = token
    props[:url] = url
    props[:solver_name] = solverName

    solver_size = props["num_qubits"] #USC DW2X = 1152, NASA 2000q = 2048
    log_solver_size = ceil(Int, solver_size/4)

    if loadInsts 
        instances = load(loadInstsFile,"instances")
        num_insts = length(instances)
    end


    eg = EmbeddingGenerator([GaugeEmbedding=>solver_size])
    hfs_eg = EmbeddingGenerator([GaugeEmbedding=>log_solver_size]) #need to create a gauge of largest size possible to work with code


    length(inst_list) == 0 && (inst_list = 1:num_insts)


    println("Starting")
    for i in inst_list  #@sync @parallel ??
        println(i, " at ",  Dates.format(now(), "HH:MM"))

        loadInsts ? (inst = instances[i]) : (inst = generatePudenzInstance(log_solver_size, numBitsUsed = problem_size, solverName = solverName))
        gs_e = false

        # HFS for finding GS Energy
        #---------------------------
        if run_hfs #&& !isfile()
            verbose && println("Running HFS")
            exper = ExperimentDescription(instance=inst, R=num_samples_hfs, name=string("instHFS_", i),solver=:HFS, solver_properties=props, 
                embedding_generator=hfs_eg, decode_opts=[1])
            (_, es, _) = run_experiment(exper, minruns=1,maxruns=num_runs_hfs,N=num_runs_hfs, vc=0,vcmode=:prob,
                savesols=false,save_encoded=false,tosave=true,savepath=dir,save_embeddings=false,coupleruns=false,timeout=2.,verbose=verbose)
            
            gs_e = minimum(es)
        end
        inst.properties[:gse] = gs_e


        if run_C

            

            # C/Repetition Code
            #-------------------
            # beta = 0 means no penalty bit (so rep code) and true means only one phys bit used (so 1 copy)
            verbose && println("Running Repetition code")
            rep_emb = PudenzEmbedding(beta = 0.0, repCode = true, problemSize = problem_size, codeName = solverName)
            inst_c = applyEmbedding(rep_emb, inst)
            exper = ExperimentDescription(instance=inst_c, R=num_samples_dw, name=string("instC_", i), solver=:DWave,
                solver_properties=props,embedding_generator=eg, decode_opts=[1])
            (_, es_dw_rep, _) = run_experiment(exper, minruns=4*num_runs_dw,maxruns=4*num_runs_dw,vc=0,vcmode=:prob, N=num_per_call,
                    savesols=true,save_encoded=false,tosave=true, 
                    save_embeddings=false,coupleruns=false,savepath=dir,timeout=2.,verbose=verbose)
            
        end


        if run_QAC
            # Pudenz Code
            #-------------
            for b in betas
                # Create the physical problem for the given beta
                p_emb = PudenzEmbedding(beta = b, codeName = solverName)
                inst_p = applyEmbedding(p_emb, inst)
                
                # Send pudenz/QAC problem off to DWave
                verbose && println("Running Pudenz Code on DWave with beta = ", b)
                exper = ExperimentDescription(instance=inst_p, R=num_samples_dw, name=string("instQAC_", i),solver=:DWave, 
                    solver_properties=props,embedding_generator=eg, decode_opts=[1])
                (_, es_dw_c, _) = run_experiment(exper, minruns=num_runs_dw,maxruns=num_runs_dw,vc=0,vcmode=:prob, N=num_per_call,
                        savesols=true,save_encoded=false,tosave=true, 
                        save_embeddings=false,coupleruns=false,savepath=dir,timeout=2.,verbose=verbose) 
            end
        end
    end

    println("Done with ", inst_list)
end


function createDataSet(; file_name::String = "instances", gen_code::Bool = false, solver_name::String = "DW2X", num_insts::Int64 = 1000, problem_size::Int64 = 288, total_size::Int64 = 288)

    instances = Array{ProblemInstance}(num_insts)

    for i = 1:num_insts
        instances[i] = generatePudenzInstance(total_size, numBitsUsed = problem_size, solverName = solver_name)
    end

    save(string(file_name, ".jld"), "instances", instances)
    nothing
end




function analysis(; dir::String = "", inst_list::Array{Int64} = [1], betas::Array{Float64} = [0.1 0.2 0.3 0.4 0.5], num_posterior_samples::Int64 = 1000)

    num_insts = length(inst_list)
    successC = Array{Float64}(num_insts, 3)
    successQAC = Array{Float64}(num_insts, 4)

    dir[end] == '/' || (dir = string(dir, '/'))

    for ix = 1:num_insts
        i = inst_list[ix]


        if isfile(string(dir, "instHFS_", i, "/1.jld2"))
            gs_e = minimum(load(string(dir, "instHFS_", i, "/1.jld2"))["decoded_energies"])
        elseif isfile(string(dir, "instC_", i, "/ExperimentDescriptions.jld2"))
            desc = load(string(dir, "instC_", i, "/ExperimentDescriptions.jld2"))["ExperimentDescriptions"]
            if typeof(desc) == Array{DWave.ExperimentDescription, 1}
                prop = desc[1].instance.properties
            else
                prop = desc["ExperimentDescriptions"][1].instance.properties
            end

            if isdefined(:prop) && haskey(prop, :gse)
                gs_e = prop[:gse]
            else
                gs_e = false
                println("Unable to find ground state energy for instance ", i, ". Will use minimum found in dwave runs")
            end
        else
            gs_e = false
            println("Unable to find ground state energy for instance ", i, ". Will use minimum found in dwave runs")
        end
            

        res = load(string(dir, "instC_", i, "/1.jld2"))
        ps = get_posteriors(res["decoded_energies"],false,num_posterior_samples,true_Egs=minimum([gs_e; res["decoded_energies"]]))[:,end]
        successC[ix, :] = [mean(ps), std(ps), gs_e]

        tempVals = [0.0 0.0 0.0]
        for j = 1:length(betas)
            res = load(string(dir, "instQAC_", i, "/", j, ".jld2"))

            ps = get_posteriors(res["decoded_energies"],false,num_posterior_samples,true_Egs=minimum([gs_e; res["decoded_energies"]]))[:,end]
            if mean(ps) > tempVals[1]
                tempVals = [mean(ps) std(ps) betas[j]]
            end
        end

        successQAC[ix, :] = [tempVals[1], tempVals[2], gs_e, tempVals[3]]
    end

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