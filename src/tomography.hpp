#include <iostream>
#include <Eigen/Dense>
//#include <random>
//#include <vector>
#include <iomanip>
#include <fstream>
//#include <bitset>

#ifndef QST_TOMOGRAPHY_HPP
#define QST_TOMOGRAPHY_HPP

namespace qst{

template<class Sampler,class Optimizer> class Tomography {

    Basis & basis_;
    Rbm & rbm_;
    Sampler & sampler_;
    Optimizer & optimizer_;

    int nv_;
    int nh_;
    int nsites_;
    int M_;
    int D_;
    int npar_;
    int bs_;
    int cd_;
    double Z_;
    double negative_log_likelihood_;
    double kl_divergence_;
    double overlap_;

    Eigen::MatrixXd grad_;
    Eigen::VectorXd deltaP_;
    Eigen::VectorXd wf_;
    
    std::mt19937 rgen_;
    
   
public:
     
    Tomography(Basis &basis,Sampler &sampler,Optimizer &optimizer,tools::Experiment &exp):basis_(basis),rbm_(sampler.Rbm()),optimizer_(optimizer),sampler_(sampler){ 
        
        std::cout<<"- Initializing tomography module"<<std::endl;
        nv_ = rbm_.Nvisible();
        nh_ = rbm_.Nhidden();
        nsites_ = basis_.Nsites();
        M_ = basis_.Nbosons();
        D_ = basis.Dimension();
        npar_=rbm_.Npar();
        bs_ = exp.bs_;
        cd_ = exp.cd_;
        
        optimizer_.SetNpar(npar_);
        grad_.resize(bs_,npar_);
        deltaP_.resize(npar_);
        rbm_.SetBatchBiases(bs_);
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
        sampler_.sample(cd_);
        for(int s=0;s<bs_;s++){
            grad_.row(s) += rbm_.DerLog(sampler_.v_.row(s));
        }
        optimizer_.getUpdates(grad_,deltaP_);
    }
    
    //Compute gradient of KL divergence 
    void GradientFE(const Eigen::MatrixXd &batch){ 
        grad_.setZero();
        Z_ = rbm_.ExactPartitionFunction();
        Eigen::VectorXd negative(npar_);
        negative.setZero(npar_);
        for(int i=0;i<D_;i++){
            negative += rbm_.amplitude(basis_.states_bin_.row(i))*rbm_.amplitude(basis_.states_bin_.row(i))*rbm_.DerLog(basis_.states_bin_.row(i))/Z_;
            //for (int p=0;p<npar_;p++){
            //    negative(p) += rbm_.amplitude(basis_states_bin_.row(i))*rbm_.amplitude(basis_states_bin_.row(i))*rbm_.DerLog(basis_states_bin_.row(i))(p)/Z_;
            //    //negative(p) += rbm_.amplitude(sample_bin)*rbm_.amplitude(sample_bin)*rbm_.DerLog(sample_bin)(p)/Z_;
            //}
        }
        for(int s=0;s<bs_;s++){
            grad_.row(s) += negative;
        }
        optimizer_.getUpdates(grad_,deltaP_);
    }

     
    //Update NN parameters
    void UpdateParameters(){
        auto pars=rbm_.GetParameters();
        optimizer_.Update(deltaP_,pars);
        rbm_.SetParameters(pars);
        rbm_.SetBatchBiases(bs_);
    }

    //Run the tomography
    void Run(Eigen::MatrixXd &trainSet,int niter,std::string &weightsName,std::ofstream &fout){
        optimizer_.Reset();
        int index,counter;
        Eigen::VectorXd sample_bin(nv_);
        Eigen::MatrixXd batch_bin(bs_,nv_);
        std::uniform_int_distribution<int> distribution(0,trainSet.rows()-1);
        int saveFrequency = 10;
        
        int ntest = 10000;
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
                    Z_ = rbm_.ExactPartitionFunction(); 
                    ExactKL();
                    Overlap();
                }
                NLL(nll_test);
                if (negative_log_likelihood_<best_nll){
                    best_overlap = overlap_;
                    best_nll = negative_log_likelihood_;
                }
                PrintStats(i,best_overlap);
                counter = 0;
            }
            counter++;
        }
    }
    
    //Compute the overlap
    void Overlap(){
        overlap_ = 0.0;
        for(int i=0;i<D_;i++){
            overlap_ += wf_(i)*rbm_.amplitude(basis_.states_bin_.row(i))/Z_;
        }
    }

    //Compute KL divergence exactly
    void ExactKL(){
        kl_divergence_ = 0.0;
        for(int i=0;i<D_;i++){
            kl_divergence_ += wf_(i)*wf_(i)*log(wf_(i)*(wf_(i)));
            kl_divergence_ += wf_(i)*wf_(i)*(log(Z_) - rbm_.LogVal(basis_.states_bin_.row(i)));
        }
    }
    
    void NLL(Eigen::MatrixXd & samples) {
        negative_log_likelihood_=0.0;
        for (int k=0;k<samples.rows();k++){
            negative_log_likelihood_ -=log(rbm_.amplitude(samples.row(k))*rbm_.amplitude(samples.row(k))/Z_);
        }
        negative_log_likelihood_ /= double(samples.rows());
    }

    //Set the value of the target wavefunction
    void setWavefunction(Eigen::VectorXd & psi){
        for(int i=0;i<D_;i++){
            wf_(i) = psi(i);
        }
    }
    
    //Print observer
    void PrintStats(int i,double best_overlap){
        std::cout << "Epoch: " << i << "\t";     
        std::cout << "KL = " << std::setprecision(10) << kl_divergence_ << "\t";
        std::cout << "NLL = " << std::setprecision(10) << negative_log_likelihood_ << "\t";
        std::cout << "Overlap = " << std::setprecision(10) << overlap_<< "\t";//<< Fcheck_;
        std::cout << "Best overlap = " << std::setprecision(10) << best_overlap;// << "\t" << Fcheck_;
        std::cout << std::endl;
    } 
};
}

#endif
