# Mixed effect neural network: Genotypes -> (complete/incomplete) Intemediate omics features -> Phenotyes


* The intermediate omics features should be in the same file as phenotypes. In this example, the "phenotypes.csv" file contains one column for the phenotype named "y1", and two columns for the omics features named "y2" and "y3". 
* Just simply indicate the named of the omics features in the `build_model()` function. In this example, we have `latent_traits=["y2","y3"]`.
* If there are many omics features (e.g., 1000), you can avoid printing the model information in the `runMCMC()` function by setting `printout_model_info=false`.
* missing omics data is allowed. Just make sure the missings are recognized in Julia. For example, if the missing values are "NA" in your raw data, then you can set `missingstrings=["NA"]` in the `CSV.read()` function. Then thouse NA will be transferred to `missing` elements in Julia. 
* You may want to set missing values manually, for example, set the phenotypes for individuals in testing dataset as missing. In julia, you should first change the type of that column to allow missing, e.g., `phenotypes[!,:y1] =  convert(Vector{Union{Missing,Float64}}, phenotypes[!,:y1])`. Then you can set missing values manually, e.g., `phenotypes[1:2,:y1] .= missing` sets the values for first two rows in column named y1 as missing.
* To include residual (e.g. not mediated by other omics features) polygenic component, you can (1) an additional hidden node in the middle layer (see example (o2)); or use a more flexible partial-connected neural network (see example (o3)).
* For the testing individuals (i.e., individuals without phenotype), if the testing individual have omics data, then incorporating those individuals in analysis will help to estimate marker effects. But if testing individuals only have genotype data, we cannot include them in our analysis. Instead, we can calculate the EBV once we have estimated the marker effects and neural network weights.


### example(o1)
```julia
# Step 1: Load packages
using JWAS,DataFrames,CSV,Statistics,JWAS.Datasets

# Step 2: Read data
phenofile  = dataset("phenotypes.csv")
genofile   = dataset("genotypes.csv")

phenotypes = CSV.read(phenofile,DataFrame,delim = ',',header=true,missingstrings=["NA"]) #should include omcis data!
genotypes  = get_genotypes(genofile,separator=',',method="BayesC")

# Step 3: Build Model Equations
model_equation  ="y1 = intercept + genotypes" #y1 is the observed phenotype
model = build_model(model_equation,
		    num_hidden_nodes=2,
                    latent_traits=["y2","y3"],  #y2 and y3 are two omics features
		    nonlinear_function="tanh")

# Step 4: Run Analysis
out = runMCMC(model,phenotypes,chain_length=5000,printout_model_info=false)

# Step 5: Check Accuruacy
results    = innerjoin(out["EBV_NonLinear"], phenotypes, on = :ID)
accuruacy  = cor(results[!,:EBV],results[!,:bv1])
```

### example(o2): NN-LMM Omics: includes a residual that is not mediated by other omics features
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

