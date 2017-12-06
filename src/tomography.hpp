#include <iostream>
#include <Eigen/Dense>
#include <iomanip>
#include <fstream>
#ifndef QST_TOMOGRAPHY_HPP
#define QST_TOMOGRAPHY_HPP

namespace qst{

template<class Sampler,class Optimizer> class Tomography {

    Rbm & rbm_;
    Sampler & sampler_;
    Optimizer & optimizer_;
    Observer & observer_;

    int npar_;
    int bs_;
    int cd_;
    int niter_;

    Eigen::VectorXd grad_;
    Eigen::VectorXd deltaP_;
    
    std::mt19937 rgen_;

public:
     
    Tomography(Sampler &sampler,Optimizer &optimizer,Observer &observer,
            tools::Parameters &par): observer_(observer),
                                     rbm_(sampler.GetRbm()),
                                     optimizer_(optimizer),
                                     sampler_(sampler){ 
        //std::cout<<"- Initializing tomography module"<<std::endl;
        npar_=rbm_.Npar();
        bs_ = par.bs_;
        cd_ = par.cd_;
        niter_ = par.ep_;
        optimizer_.SetNpar(npar_);
        grad_.resize(npar_);
        deltaP_.resize(npar_);
        sampler_.Reset(true);
    }

    //Compute gradient of KL divergence 
    void Gradient(const Eigen::MatrixXd &batch){ 
        grad_.setZero();
        sampler_.Reset();   
        //Positive Phase driven by the data
        for(int s=0;s<bs_;s++){
            grad_ -= rbm_.DerLog(batch.row(s))/double(bs_);
        }
        sampler_.Sample(cd_);
        for(int s=0;s<sampler_.Nchains();s++){
            grad_ += rbm_.DerLog(sampler_.VisibleStateRow(s))/double(sampler_.Nchains());//sampler_.v_.row(s));
        }
        optimizer_.getUpdates(grad_,deltaP_);
    }

    //Update NN parameters
    void UpdateParameters(){
        auto pars=rbm_.GetParameters();
        optimizer_.Update(deltaP_,pars);
        rbm_.SetParameters(pars);
    }

    //Run the tomography
    void Run(Eigen::MatrixXd &trainSet,std::string &weightsName,std::ofstream &fout){
        // Initialization
        int index;
        int saveFrequency = 10;
        int counter = 0;
        int ntest = 1000;
        Eigen::MatrixXd batch_bin;
        Eigen::VectorXd sample_bin(rbm_.Nvisible());
        Eigen::MatrixXd nll_test(ntest,rbm_.Nvisible());
        std::uniform_int_distribution<int> distribution(0,trainSet.rows()-1);
        optimizer_.Reset();

        // Generate test-set of negative log-likelihood
        for (int i=0;i<ntest;i++){
            index = distribution(rgen_);
            tools::MultinomialToBinomial(rbm_.Nsites(),rbm_.MaxNbosons(),trainSet.row(index),sample_bin);
            nll_test.row(i) = sample_bin;
        }
       
        //Training loop
        for(int i=0;i<niter_;i++){
            //Build the initial state for the sampler
            batch_bin.resize(sampler_.Nchains(),rbm_.Nvisible());
            for(int k=0;k<sampler_.Nchains();k++){
                index = distribution(rgen_);
                tools::MultinomialToBinomial(rbm_.Nsites(),rbm_.MaxNbosons(),trainSet.row(index),sample_bin); 
                batch_bin.row(k) = sample_bin;
            }
            sampler_.SetVisibleLayer(batch_bin);
            
            // Build the batch of data from a random permutation
            batch_bin.resize(bs_,rbm_.Nvisible()); 
            for(int k=0;k<bs_;k++){
                index = distribution(rgen_);
                tools::MultinomialToBinomial(rbm_.Nsites(),rbm_.MaxNbosons(),trainSet.row(index),sample_bin);
                batch_bin.row(k) = sample_bin;
            }
            
            // Perform one step of optimization
            Gradient(batch_bin);
            UpdateParameters();
            //Compute stuff and print
            if (counter == saveFrequency){
                //rbm_.PrintWeights();
                //observer_.Scan(i,nll_test);
                observer_.ComparePartitionFunctions(i);
                counter = 0;
            }
            counter++;
        }
    }
};
}

#endif
