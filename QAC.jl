
using DWave, JLD, Plots


#Changes to implement:
# -store/retreive embedding code files within storage directory
# -create more plots
# -option to create, store, and retreive set of instances

function runQAC(token ; solverName="DW2X", url="https://usci.qcc.isi.edu/sapi", #solver info
    dir="test/", #loadInsts = false, loadInstsFile = "",#storage info
    problem_size=288, num_insts=5, betas=[0.1, 0.2, 0.3, 0.4, 0.5], #QAC vs. C info
    num_per_call=10,  num_runs_dw=20, num_samples_dw=1000, num_runs_hfs=20, use_hfs=true, num_samples_hfs=1000, num_posterior_samples=1000) #experiment info

    #Create the directory if it doesn't exist
    dir[end] == '/' || (dir = string(dir, '/'))
    mkpath(dir)

    solv = DWave.dwrm.RemoteConnection(url, token)[:get_solver](solverName)
    isfile(string(solverName, "_code.jld")) || generatePudenzCode(solver=solv, codeName=solverName)
    props = solv[:properties]
    props[:token] = token
    props[:url] = url
    props[:solver_name] = solverName

    solver_size = props["num_qubits"] #USC DW2X = 1152

    #if loadInsts
    #    file = JLD.jldopen(loadInstsFile, "r")
    #    instances = read(file)["instances"]

    #    close(file)
    #end

    successQAC = zeros(num_insts, 4)
    successC = zeros(num_insts, 3)

    eg = EmbeddingGenerator([GaugeEmbedding=>solver_size])
    hfs_eg = EmbeddingGenerator([GaugeEmbedding=>ceil(Int, solver_size/4)]) #need to create a gauge of largest size possible to work with code


    println("Starting")
    for i = 1:num_insts

        #if loadInsts
        #    inst = instances[i]
        #else
        inst = generatePudenzInstance(solver_size/4, numBitsUsed = problem_size, solverName = solverName)
        #end


        # HFS for finding GS Energy
        #---------------------------
        if use_hfs
            println("Running HFS")
            exper = ExperimentDescription(instance=inst, R=num_samples_hfs, name=string("instHFS_", i),solver=:HFS, solver_properties=props, 
                embedding_generator=hfs_eg, decode_opts=[1])
            (_, es, _) = run_experiment(exper, minruns=1,maxruns=num_runs_hfs,vc=0,vcmode=:prob, N=1,
                savesols=false,save_encoded=false,tosave=false, save_embeddings=false,coupleruns=false,timeout=2.,verbose=true)
            
            gs_e = minimum(es)
        else
            gs_e = false
        end


        # C/Repetition Code
        #-------------------
        # beta = 0 means no penalty bit (so rep code) and true means only one phys bit used (so 1 copy)
        println("Running Repetition code")
        rep_emb = PudenzEmbedding(beta = 0.0, repCode = true, problemSize = problem_size, codeName = solverName)
        inst_c = applyEmbedding(rep_emb, inst)
        exper = ExperimentDescription(instance=inst_c, R=num_samples_dw, name=string("instC_", i), solver=:DWave,
            solver_properties=props,embedding_generator=eg, decode_opts=[1])
        (_, es_dw, _) = run_experiment(exper, minruns=4*num_runs_dw,maxruns=4*num_runs_dw,vc=0,vcmode=:prob, N=num_per_call,
                savesols=true,save_encoded=false,tosave=true, 
                save_embeddings=false,coupleruns=false,savepath=dir,timeout=2.,verbose=true)
        
        ps = get_posteriors(es_dw,false,num_posterior_samples,true_Egs=gs_e)[:,end]
        successC[i, :] = [mean(ps), std(ps), gs_e]
        
        
        
        # Pudenz Code
        #-------------
        tempVals = zeros(3)
        for b in betas
            # Create the physical problem for the given beta
            p_emb = PudenzEmbedding(beta = b, codeName = solverName)
            inst_p = applyEmbedding(p_emb, inst)
            
            # Send pudenz/QAC problem off to DWave
            println("Running Pudenz Code on DWave with beta = ", b)
            exper = ExperimentDescription(instance=inst_p, R=num_samples_dw, name=string("instQAC_", i),solver=:DWave, 
                solver_properties=props,embedding_generator=eg, decode_opts=[1])
            (_, es_dw, _) = run_experiment(exper, minruns=num_runs_dw,maxruns=num_runs_dw,vc=0,vcmode=:prob, N=num_per_call,
                    savesols=true,save_encoded=false,tosave=true, 
                    save_embeddings=false,coupleruns=false,savepath=dir,timeout=2.,verbose=true)
            
            # Select the best results
            ps = get_posteriors(es_dw,false,num_posterior_samples,true_Egs=gs_e)[:,end]
            if mean(ps) > tempVals[1]
                tempVals = [mean(ps) std(ps) b]
            end
        end
        
        successQAC[i, :] = [tempVals[1], tempVals[2], gs_e, tempVals[3]]
    end

    println("Done")

    #Can all this just be gotten from what's already stored?
    save(string(dir,"success_QAC.jld"), "mean", successQAC[:, 1])
    save(string(dir,"success_QAC.jld"), "std", successQAC[:, 2])
    save(string(dir,"success_QAC.jld"), "gs_e", successQAC[:, 3])
    save(string(dir,"success_QAC.jld"), "betas", successQAC[:, 4])

    save(string(dir,"success_C.jld"), "mean", successC[:, 1])
    save(string(dir,"success_C.jld"), "std", successC[:, 2])
    save(string(dir,"success_C.jld"), "gs_e", successC[:, 3])


    # Results
    #---------
    scatter(successQAC[:, 1], successC[:, 1], m = ColorGradient(:rainbow), zcolor = successQAC[:, 4])
    png(string(dir, "results"))
end