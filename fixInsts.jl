#Will fix any issues that occured while decoding solutions/reverse embedding
using DWave, JLD

function fix(dir, inst_list; betas = [0.1, 0.2, 0.3, 0.4, 0.5], codeName = "DW2X")

	instances = load(string(dir, "/instances.jld"),"instances")

	for i in inst_list
		inst = instances[i]

		for b = 1:length(betas)
			try
				res = load(string(dir, "/instQAC_", i, "/", b, ".jld2"))
				if size(res["decoded_solutions"], 1) != length(inst.h)

					(numSamps, numReads) = size(res["energies"])
					newE = Array{Float64}(numSamps, numReads)
					newSols = BitArray(length(inst.h), numSamps, numReads)
					pemb = PudenzEmbedding(beta = betas[b], codeName = codeName)
					inst_Q = applyEmbedding(pemb, inst)

					for j = 1:numReads
						(newE[:, j], newSols[:, :, j]) = reverseEmbedding(pemb, inst_Q, res["decoded_solutions"][:, :, j])
					end

					res["decoded_solutions"] = newSols
					res["decoded_energies"] = newE
					save(string(dir, "/instQAC_", i, "/", b, ".jld2"), res)
				end
			catch e
				println(e)
				println("This occurred while trying to work on: ", string(dir, "/instQAC_", i, "/", b, ".jld2"))
			end
		end
	end
end