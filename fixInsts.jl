#Will fix any issues that occured while decoding solutions/reverse embedding
using DWave, JLD

function fix(dir::String, inst_list::Array{Int64}; betas::Array{Float64} = [0.1, 0.2, 0.3, 0.4, 0.5], codeName::String = "DW2X", fix_C::Bool = true, fix_QAC::Bool = true)

	instances = load(string(dir, "/instances.jld"),"instances")
	#The C/Rep Code unembedding is the same for all instances
	fix_C && (cemb = PudenzEmbedding(beta = 0.0, repCode = true, codeName = codeName))

	for i in inst_list
		inst = instances[i]


		#Fix C Instances
		if fix_C
			try
				res = load(string(dir, "/instC_", i, "/1.jld2"))

				#Checking to be sure this instance hasn't been unembedded
				if size(res["decoded_solutions"], 1) != length(inst.h)

					(numSamps, numRuns) = size(res["energies"])
					newE = Array{Float64}(numSamps, numRuns)
					newSols = BitArray(length(inst.h), numSamps, numRuns)
					inst_tmp = applyEmbedding(cemb, inst)

					for j = 1:numRuns					
						(newE[:, j], newSols[:, :, j]) = reverseEmbedding(cemb, inst_tmp, res["decoded_solutions"][:, :, j])
					end

					res["decoded_solutions"] = newSols
					res["decoded_energies"] = newE
					save(string(dir, "/instC_", i, "/1.jld2"), res)
				end
			catch e
				println(e)
				println("This occurred while trying to work on: ", string(dir, "/instC_", i, "/1.jld2\n"))
			end
		end


		#Fix QAC Instances
		if fix_QAC
			for b = 1:length(betas)
				try
					res = load(string(dir, "/instQAC_", i, "/", b, ".jld2"))

					#Checking to be sure this instance hasn't been unembedded
					if size(res["decoded_solutions"], 1) != length(inst.h)

						(numSamps, numRuns) = size(res["energies"])
						newE = Array{Float64}(numSamps, numRuns)
						newSols = BitArray(length(inst.h), numSamps, numRuns)
						pemb = PudenzEmbedding(beta = betas[b], codeName = codeName)
						inst_tmp = applyEmbedding(pemb, inst)

						for j = 1:numRuns						
							(newE[:, j], newSols[:, :, j]) = reverseEmbedding(pemb, inst_tmp, res["decoded_solutions"][:, :, j])
						end

						res["decoded_solutions"] = newSols
						res["decoded_energies"] = newE
						save(string(dir, "/instQAC_", i, "/", b, ".jld2"), res)
					end
				catch e
					println(e)
					println("This occurred while trying to work on: ", string(dir, "/instQAC_", i, "/", b, ".jld2\n"))
				end
			end
		end
	end
end