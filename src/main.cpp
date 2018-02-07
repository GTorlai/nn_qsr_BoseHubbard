#include <iostream>
#include "qst.hpp"
#include <mpi.h>

int main(int argc, char* argv[]){
   
    //---- MPI ----// 
    //MPI_Init(&argc,&argv);
    //int rank;
    //MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    //int nPROC;
    //MPI_Comm_size(MPI_COMM_WORLD,&nPROC);
    
    std::string command = argv[1];
    if ((command == "-h") || (command == "-help")){
        //printHeader();
        exit(0);
    }

    //---- PARAMETERS ----//
    tools::Parameters par;
    //Read simulation parameters from command line
    par.ReadParameters(argc,argv);

    //---- OPTIMIZER ----//
    // Stochastic gradient descent
    typedef qst::Sgd Optimizer;
    Optimizer optimizer(par.lr_,0.0,0.0,0.0);
    par.opt_="sgd";
    // Ada Delta
    //typedef qst::AdaDelta Optimizer;
    //Optimizer optimizer(0.90,1.0e-6);
    //par.opt_="adaDel";
    
    //---- BASIS ----//
    qst::Basis basis(par.nsites_,par.M_,par.M_max_);
    //basis.PrintInfos();
    
    //---- RBM ----//
    qst::Rbm rbm(par);
    rbm.InitRandomPars(12345,par.w_);
    //rbm.PrintWeights();
    
    //---- SAMPLER ----//
    //Gibbs sampling
    typedef qst::Gibbs Sampler;
    Sampler sampler(rbm,par.nsites_,par.M_max_,par.nc_);
    par.alg_="CD";
    // Parallel Tempering
    //typedef qst::ParallelTempering Sampler;
    //Sampler sampler(rbm,par.nsites_,par.M_max_,par.nc_,par.nrep_);
    //par.alg_="PT";

    //---- OBSERVER ----//
    qst::ParallelTempering pt(rbm,par.nsites_,par.M_max_,par.nc_,par.nrep_);
    qst::AIS ais(rbm,par.nsites_,par.M_max_,par.nc_,par.nrep_);
    qst::Observer observer(basis,rbm,pt,ais);
    
    //---- TOMOGRAPHY ----//
    qst::Tomography<Sampler,Optimizer> tomo(sampler,optimizer,observer,par);  

    ////---- Data handling ----//
    par.SetNumberVisible(rbm.Nvisible());
    Eigen::MatrixXd training_samples;
    Eigen::MatrixXd test_samples;
    Eigen::VectorXd target_state;
    std::string network_name;
    tools::LoadTrainingData(par,training_samples);
    tools::LoadWavefunction(par,target_state,basis.Dimension());
    network_name = tools::NetworkName(par);
    std::ofstream prova(tools::RbmObserverName(par));
    observer.setWavefunction(target_state);
    
    //---- Execute ----//
    tomo.Run(training_samples,network_name,prova); 


    //MPI_Finalize();
}
