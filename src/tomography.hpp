#include <iostream>
#include <Eigen/Dense>
#include <iomanip>
#include <fstream>
#ifndef QST_TOMOGRAPHY_HPP
#define QST_TOMOGRAPHY_HPP

namespace qst{

template<class Sampler,class Optimizer> class Tomography {

    Basis & basis_;
    Rbm & rbm_;
    Sampler & sampler_;
    Optimizer & optimizer_;
    Observer & observer_;

    int nv_;
    int nh_;
    int nsites_;
    int M_;
    int D_;
    int npar_;
    int bs_;
    int cd_;
    int nc_;
    double Z_;
    double negative_log_likelihood_;
    double kl_divergence_;
    double overlap_;
    Eigen::MatrixXd grad_;
    Eigen::VectorXd deltaP_;
    Eigen::VectorXd wf_;
    
    std::mt19937 rgen_;

public:
     
    Tomography(Basis &basis,Sampler &sampler,Optimizer &optimizer,
               tools::Parameters &par,Observer &observer):
        basis_(basis),rbm_(sampler.Rbm()),
        optimizer_(optimizer),sampler_(sampler),
        observer_(observer){ 
        
        std::cout<<"- Initializing tomography module"<<std::endl;
        nv_ = rbm_.Nvisible();
        nh_ = rbm_.Nhidden();
        nsites_ = basis_.Nsites();
        M_ = basis_.Nbosons();
        D_ = basis_.Dimension();
        npar_=rbm_.Npar();
        bs_ = par.bs_;
        cd_ = par.cd_;
        nc_ = par.nc_;

        optimizer_.SetNpar(npar_);
        grad_.resize(bs_,npar_);
        deltaP_.resize(npar_);
        rbm_.SetBatchBiases(nc_);
    }

    //Compute gradient of KL divergence 
    void GradientCD(const Eigen::MatrixXd &batch){ 
        grad_.setZero();
        //Positive Phase driven by the data
        for(int s=0;s<bs_;s++){
            grad_.row(s) -= rbm_.DerLog(batch.row(s));
        }
        //Negative Phase driven by the model
        sampler_.SetVisibleLayer(batch);
        sampler_.Sample(cd_);
        for(int s=0;s<bs_;s++){
            grad_.row(s) += rbm_.DerLog(sampler_.VisibleStateRow(s));//sampler_.v_.row(s));
        }
        optimizer_.getUpdates(grad_,deltaP_);
    }
    
    //Compute gradient of KL divergence 
    void GradientFE(const Eigen::MatrixXd &batch){ 
        grad_.setZero();
        Z_ = rbm_.ExactPartitionFunction(basis_);
        Eigen::VectorXd negative(npar_);
        negative.setZero(npar_);
        for(int i=0;i<D_;i++){
            negative += rbm_.amplitude(basis_.states_bin_.row(i))*rbm_.amplitude(basis_.states_bin_.row(i))*rbm_.DerLog(basis_.states_bin_.row(i))/Z_;
        }
        for(int s=0;s<bs_;s++){
            grad_.row(s) -= rbm_.DerLog(batch.row(s));
            grad_.row(s) += negative;
        }
        optimizer_.getUpdates(grad_,deltaP_);
    }

     
    //Update NN parameters
    void UpdateParameters(){
        auto pars=rbm_.GetParameters();
        optimizer_.Update(deltaP_,pars);
        rbm_.SetParameters(pars);
        rbm_.SetBatchBiases(nc_);
    }

    //Run the tomography
    void Run(Eigen::MatrixXd &trainSet,int niter,std::string &weightsName,std::ofstream &fout){
        optimizer_.Reset();
        int index,counter;
        Eigen::VectorXd sample_bin(nv_);
        Eigen::MatrixXd batch_bin(bs_,nv_);
        std::uniform_int_distribution<int> distribution(0,trainSet.rows()-1);
        int saveFrequency = 10;
        
        int ntest = 100;
        double best_overlap=0.0;
        double best_nll=1000.0;

        Eigen::MatrixXd nll_test(ntest,nv_);
        for (int i=0;i<ntest;i++){
            index = distribution(rgen_);
            tools::MultinomialToBinomial(nsites_,M_,trainSet.row(index),sample_bin);
            nll_test.row(i) = sample_bin;
        }
        
        counter = 0;
        for(int i=0;i<niter;i++){
            for(int k=0;k<bs_;k++){
                index = distribution(rgen_);
                tools::MultinomialToBinomial(nsites_,M_,trainSet.row(index),sample_bin);
                batch_bin.row(k) = sample_bin;
            }
            GradientCD(batch_bin);
            UpdateParameters();
            
            if (counter == saveFrequency){
                if (nsites_<10){
                    Z_ = rbm_.ExactPartitionFunction(basis_); 
                    observer_.ExactKL(Z_);
                    observer_.Overlap(Z_);
                    observer_.NLL(nll_test,Z_);
                    if (observer_.negative_log_likelihood_<best_nll){
                        best_overlap = observer_.overlap_;
                        best_nll = observer_.negative_log_likelihood_;
                    }
                    observer_.PrintStats(i,best_overlap);
                    counter = 0;
                }
                else{
                    std::cout<<"Epoch: "<<i<<std::endl;
                }

            }
            counter++;
        }
    }
};
}

#endif
