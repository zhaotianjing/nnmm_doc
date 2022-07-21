################## Pig Genotypes ##################

#the raw pig genotypes data (3534 inds, 52843 SNPs) can be downloaded from:
    # Matthew A Cleveland, John M Hickey, Selma Forni, A Common Dataset for Genomic
    # Analysis of Livestock Populations, G3 Genes|Genomes|Genetics, Volume 2, Issue 4,
    # 1 April 2012, Pages 429–435, https://doi.org/10.1534/g3.111.001453
#After QC to remove SNPs with MAF<0.01 and fixed SNPs, we have 50436 SNPs remains
#saved in a file named "geno_n3534_p50436.csv".
cd("/Users/tianjing/nnmm_doc/data_simulation")

using DataFrames,CSV,Statistics,DelimitedFiles,Random, LinearAlgebra, StatsBase

geno_path="/Users/tianjing/Library/CloudStorage/Box-Box/singlestepdata/pig_cleveland/geno_n3534_p50436.csv"
geno_all=CSV.read(geno_path,DataFrame) # 3534 inds, 50436 SNPs
geno_all_matrix=Matrix(geno_all[:,2:end])

Random.seed!(123)
#only select 100 individuals and 200 SNPs
ind_pos=sample(1:3534, 100,replace=false, ordered=true)
snp_pos=sample(1:50436,200,replace=false, ordered=true)
geno=geno_all_matrix[ind_pos,snp_pos]  #100 ind, 200 SNPs
# writedlm("geno_n100_p200.txt",geno)
geno_df = DataFrame(geno,["m$i" for i in 1:200])
insertcols!(geno_df,1,:ID => 1:100)
CSV.write("geno_n100_p200.csv",geno_df)


################## Simulated 10 Omics ##################
#read genotypes
geno=CSV.read("geno_n100_p200.csv",DataFrame)
geno=Matrix(geno[:,2:end])

nOmics=10
nInd=100

σ2_mi=2               #as in Christensenet al.(2021)
h2_m=0.61             #as in Christensenet al.(2021)
σ2_gi=h2_m*σ2_mi      #genetic variance of each omics
σ2_ei=(1-h2_m)*σ2_mi  #residual variance of each omics

#select QTL for each omicc, assume each omics is affected by 20 QTLs
#also simulate QTL effects
Random.seed!(1)
QTL_select_pos    = Int.(zeros(20,nOmics))
QTL_select_effect = zeros(20,nOmics)
for o in 1:nOmics
    QTL_select_pos[:,o]    = sample(1:200, 20,replace=false, ordered=true)
    QTL_select_effect[:,o] = randn(20)
end

#simulate omics
G=zeros(nInd,nOmics) #genetic value of omics
E=zeros(nInd,nOmics) #residuals of omics
Random.seed!(1)
for i in 1:nOmics
	W=geno[:,QTL_select_pos[:,i]]
	@show size(W),W[1:2,1:2]
    Gi=W*QTL_select_effect[:,i]
	Gi=Gi/std(Gi)*sqrt(σ2_gi)
	G[:,i]=Gi
	E[:,i]=randn(nInd)*sqrt(σ2_ei)
end

var(G,dims=1)
var(E,dims=1)

# writedlm("omics_bv.txt",G)
# writedlm("omics_residual.txt",E)

omics=G+E
# writedlm("omics.txt",omics)
var(omics,dims=1)
omics_df=DataFrame(omics,["omics$i" for i in 1:10])
insertcols!(omics_df,1,:ID=>1:100)
CSV.write("omics.csv",omics_df)

################## simulate neural network weight between omics and phenotype (omics effects on phenotype) ##################
w1=randn(nOmics)
# writedlm("w1.txt",w1)


################## simulate phenotypes ##################
mysigmoid(x) = 1/(1+exp(-x))
#save g(omics_g)
G_nonlinear=mysigmoid.(G)
# writedlm("omics_bv_nonlinear.txt",G_nonlinear)
#save omics nonlinear
omics_nonlinear=mysigmoid.(omics)
# writedlm("omics_nonlinear.txt",omics_nonlinear)

#save true bv
tbv=G_nonlinear*w1
# writedlm("truebv.txt",tbv)

#create y_nonlinear
omics_contribution=omics_nonlinear*w1
σ2_z=var(omics_contribution)
σ2_g_nonlinear=var(G_nonlinear*w1)
#h2=σ2_g_nonlinear/(σ2_z+σ2_e)
h2=0.337 #as in Christensenet al.(2021)
σ2_e=σ2_g_nonlinear/h2 - σ2_z #residual variance of y

Random.seed!(1)
y_residual=randn(nInd)*sqrt(σ2_e)
var(y_residual)
y_nonlinear=omics_contribution+y_residual
var(y_nonlinear)
σ2_g_nonlinear/var(y_nonlinear)
# writedlm("y.txt",y_nonlinear)

y_df=DataFrame(ID=1:100,y=y_nonlinear,bv=tbv)
CSV.write("y.csv",y_df)

