# Partial-connected neural networks

In partial-connected neural networks, SNPs can be divided into groups by users, and each group connects to its own intermediate trait in the middle layer.  Genotype groups should be loaded seperatly.

## example(c): partial-connected neural networks, all intemediate traits are unobserved
- nonlinear function (to define relationship between middle layer and phenotye): tanh
- number of nodes in the middle layer: 3
- Bayesian model (to sample marker effects on intemediate traits): 
  - genotype group 1: single-trait BayesA
  - genotype group 2: single-trait BayesB
  - genotype group 3: single-trait BayesC
- sample the unobserved intemediate traits in the middle layer: Hamiltonian Monte Carlo

![](https://github.com/zhaotianjing/figures/blob/main/partial_example.png)

```julia
# Step 1: Load packages
using JWAS,DataFrames,CSV,Statistics,JWAS.Datasets,Random
Random.seed!(123)

# Step 2: Read data
phenofile   = dataset("phenotypes.csv")       
genofile1   = dataset("genotypes_group1.csv")  #path of genotype group1
genofile2   = dataset("genotypes_group2.csv")  #path of genotype group2
genofile3   = dataset("genotypes_group3.csv")  #path of genotype group3

phenotypes = CSV.read(phenofile,DataFrame,delim = ',',header=true,missingstrings=["NA"])
geno1  = get_genotypes(genofile1,separator=',',method="BayesA");   #read genotype group1
geno2  = get_genotypes(genofile2,separator=',',method="BayesB");   #read genotype group2
geno3  = get_genotypes(genofile3,separator=',',method="BayesC");   #read genotype group3

# Step 3: Build Model Equations
model_equation = "y1 = intercept + geno1 + geno2 + geno3";  #middle node1=intercept + geno1
                                                            #middle node2=intercept + geno2
                                                            #middle node3=intercept + geno3
model = build_model(model_equation,nonlinear_function="tanh")

# Step 4: Run Analysis
out = runMCMC(model, phenotypes, chain_length=5000);

# Step 5: Check Accuruacy
results    = innerjoin(out["EBV_NonLinear"], phenotypes, on = :ID)
accuruacy  = cor(results[!,:EBV],results[!,:bv1])
```


## example(o3): partial-connected neural networks with omics data


* Same as for fully-connected neural network, the names of omics features should be put in the `build_model()` function through the `latent_traits` argument. The order of omics and the order of genotype groups in the model equation should be consistant.

In below example, we assume genotype group 1 only affect omics 1, and genotype group 2 only affect omics 2.

![](https://github.com/zhaotianjing/figures/blob/main/part4_partial_omics.png)

```julia
# Step 1: Load packages
using JWAS,DataFrames,CSV,Statistics,JWAS.Datasets,Random
Random.seed!(1)

# Step 2: Read data
phenofile   = "/Users/tianjing/nnmm_doc/data_simulation/y.csv"
omicsfile   = "/Users/tianjing/nnmm_doc/data_simulation/omics.csv"
genofile1   = "/Users/tianjing/nnmm_doc/data_simulation/geno_group1.csv"
genofile2   = "/Users/tianjing/nnmm_doc/data_simulation/geno_group2.csv"
genofile3   = "/Users/tianjing/nnmm_doc/data_simulation/geno_group3.csv"
genofile4   = "/Users/tianjing/nnmm_doc/data_simulation/geno_group4.csv"
genofile5   = "/Users/tianjing/nnmm_doc/data_simulation/geno_group5.csv"

phenotypes  = CSV.read(phenofile,DataFrame)
omics       = CSV.read(omicsfile,DataFrame)[:,1:6] #only use first 5 omics for demonstration
omics_names = names(omics)[2:end] #get names of omics
insertcols!(omics,2,:y => phenotypes[:,:y], :bv => phenotypes[:,:bv]) #phenotype and omics should be in the same dataframe

geno1  = get_genotypes(genofile1,separator=',',method="BayesA");
geno2  = get_genotypes(genofile2,separator=',',method="BayesB");
geno3  = get_genotypes(genofile3,separator=',',method="BayesC");
geno4  = get_genotypes(genofile4,separator=',',method="RR-BLUP");
geno5  = get_genotypes(genofile5,separator=',',method="BayesL");

# Step 3: Build Model Equations
model_equation = "y = intercept + geno1 + geno2 + geno3 + geno4 + geno5"; #omics1=intercept + geno1; omics2=intercept + geno2; ...
model = build_model(model_equation,
		            num_hidden_nodes=5,
		            nonlinear_function="sigmoid",
	                latent_traits=omics_names)

# Step 4: Run Analysis
out = runMCMC(model, omics, chain_length=5000, printout_model_info=false);

# Step 5: Check Accuruacy
results    = innerjoin(out["EBV_NonLinear"], omics, on = :ID)
accuruacy  = cor(results[!,:EBV],results[!,:bv])
```
