# Mixed effect neural network: Genotypes -> (complete/incomplete) Intemediate omics features -> Phenotyes

Tips:
* Put the names of omics features in the `build_model()` function through the `latent_traits` argument.
* If there are many omics features (e.g., 1000), it is recommanded to set `printout_model_info=false` in the `runMCMC()` function.
* Missing omics data for individuals in the training dataset (i.e., individuals with phenotypes) is allowed. When you read a file with missing values via the `CSV.read()` function, the `missingstrings` argument should be used to set sentinel values that will be parsed as `missing`.
* For individuals in the testing dataset (i.e., individuals without phenotypes), if the testing individuals have complete omics data, then incorporating the omics data of those individuals may improve the relationship between input layer (genotype) and middle layer (omics).


## example(o1): fully-connected neural networks with observed intemediate omics features
* nonlinear function (to define relationship between omics and phenotye): sigmoid (other supported activation functions: "tanh", "relu", "leakyrelu", "linear")
* number of omics features in the middle layer: 10
* Bayesian model: multiple independent single-trait BayesC (to sample marker effects on intemediate omics)
* sample the missing omics in the middle layer: Hamiltonian Monte Carlo

![](https://github.com/zhaotianjing/figures/blob/main/part3_example.png)

```julia
# Step 1: Load packages
using JWAS,DataFrames,CSV,Statistics,JWAS.Datasets, Random, HTTP #HTTP to download demo data from github
Random.seed!(123)

# Step 2: Read data (from github)
phenofile  = HTTP.get("https://raw.githubusercontent.com/zhaotianjing/nnmm_doc/main/data_simulation/y.csv").body
omicsfile  = HTTP.get("https://raw.githubusercontent.com/zhaotianjing/nnmm_doc/main/data_simulation/omics.csv").body
genofile   = HTTP.get("https://raw.githubusercontent.com/zhaotianjing/nnmm_doc/main/data_simulation/geno_n100_p200.csv").body
phenotypes = CSV.read(phenofile,DataFrame)
omics      = CSV.read(omicsfile,DataFrame)
geno_df    = CSV.read(genofile,DataFrame)

omics_names = names(omics)[2:end]
insertcols!(omics,2,:y => phenotypes[:,:y], :bv => phenotypes[:,:bv])
genotypes = get_genotypes(geno_df,separator=',',method="BayesC")

# Step 3: Build Model Equations
model_equation  ="y = intercept + genotypes"
model = build_model(model_equation,
		    num_hidden_nodes=10,
                    latent_traits=omics_names,
		    nonlinear_function="sigmoid")

# Step 4: Run Analysis
out=runMCMC(model, omics, chain_length=5000, printout_model_info=false);

# Step 5: Check Accuruacy
results    = innerjoin(out["EBV_NonLinear"], omics, on = :ID)
accuruacy  = cor(results[!,:EBV],results[!,:bv])
```


### example(o2): includes a residual that is not mediated by other omics features
* To include residuals polygenic component (i.e., directly from genotypes to phenotypes, not mediated by omics features

![image](https://user-images.githubusercontent.com/18593116/180110202-f4554178-1503-4b2b-a969-c92977160540.png)
, you can (1) an additional hidden node in the middle layer (see example (o2)); or use a more flexible partial-connected neural network (see example (o3)).

This can be done by adding an extra hidden node. For all individuals, this extra hidden node will be treated as unknown to be sampled.

The example for fully-connected neural network and partial-connected neural network:

![](https://github.com/zhaotianjing/figures/blob/main/wiki_omics_residual.png)


Example code for fully-connected neural network with residual:
```julia
# Step 1: Load packages
using JWAS,DataFrames,CSV,Statistics,JWAS.Datasets

# Step 2: Read data
phenofile  = dataset("phenotypes.csv")
genofile   = dataset("genotypes.csv")
phenotypes = CSV.read(phenofile,DataFrame,delim = ',',header=true,missingstrings=["NA"])
insertcols!(phenotypes, 5, :residual => missing)  #add one column named "residual" with missing values, position is the 5th column in phenotypes
phenotypes[!,:residual] = convert(Vector{Union{Missing,Float64}}, phenotypes[!,:residual]) #transform the datatype is required for Julia
genotypes  = get_genotypes(genofile,separator=',',method="BayesC")

# Step 3: Build Model Equations
model_equation  ="y1 = intercept + genotypes"   #y1 is the observed phenotype
model = build_model(model_equation,
		    num_hidden_nodes=3,
                    latent_traits=["y2","y3","residual"],  #y2 and y3 are two omics features
		    nonlinear_function="tanh")

# Step 4: Run Analysis
out = runMCMC(model,phenotypes,chain_length=5000,printout_model_info=false)

# Step 5: Check Accuruacy
results    = innerjoin(out["EBV_NonLinear"], phenotypes, on = :ID)
accuruacy  = cor(results[!,:EBV],results[!,:bv1])
```

Example code for partial-connected neural network with residual:
```julia
# Step 1: Load packages
using JWAS,DataFrames,CSV,Statistics,JWAS.Datasets

# Step 2: Read data
phenofile   = dataset("phenotypes.csv")
genofile1   = dataset("genotypes_group1.csv")
genofile2   = dataset("genotypes_group2.csv")
genofile3   = dataset("GRM.csv")

phenotypes = CSV.read(phenofile,DataFrame,delim = ',',header=true,missingstrings=["NA"])
insertcols!(phenotypes, 5, :residual => missing)  #add one column named "residual" with missing values, position is the 5th column in phenotypes
phenotypes[!,:residual] = convert(Vector{Union{Missing,Float64}}, phenotypes[!,:residual]) #transform the datatype is required for Julia

geno1  = get_genotypes(genofile1,separator=',',method="BayesA");
geno2  = get_genotypes(genofile2,separator=',',method="BayesC");
geno3  = get_genotypes(genofile3,separator=',',header=false,method="GBLUP");

# Step 3: Build Model Equations
model_equation = "y1 = intercept + geno1 + geno2 + geno3";
model = build_model(model_equation,
		    num_hidden_nodes=3,
		    nonlinear_function="tanh",
	            latent_traits=["y3","y2","residual"])

# Step 4: Run Analysis
out = runMCMC(model, phenotypes, chain_length=5000,printout_model_info=false);

# Step 5: Check Accuruacy
results    = innerjoin(out["EBV_NonLinear"], phenotypes, on = :ID)
accuruacy  = cor(results[!,:EBV],results[!,:bv1])
```


### Julia Tips:
* You may want to set missing values manually, for example, set the phenotypes for individuals in testing dataset as missing. In julia, you should first change the type of that column to allow missing, e.g., `phenotypes[!,:y1] =  convert(Vector{Union{Missing,Float64}}, phenotypes[!,:y1])`. Then you can set missing values manually, e.g., `phenotypes[1:2,:y1] .= missing` sets the values for first two rows in column named y1 as missing.
