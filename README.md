# QAC

This is an experiment to compare QAC Pudenz codes and repetition codes that use the same number of resources (4 physical bits for 1 logical bit) for any DWave.

Primary usage (in julia repl or file):
  include("QAC.jl")
  runQAC("API Token", url="dwave sapi url", solver = "name of solver to be used", problem_size = "size of problem to be tested... at most 1/4 the size of the physical graph", ...)
  
The API Token must be provided and must be the first argument like in the example above. Otherwise, all arguments are optional and can be given in any order since they are keywords. The code should make the default experiment rather clear.

This experiment relies on the methods for QAC in DWave.jl found at https://github.com/joshjob42/DWave.jl. The key pieces are QAC.jl, embeddings.jl, and experiments.jl in the src folder.
