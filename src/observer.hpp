#ifndef QST_OBSERVER_HPP
#define QST_OBSERVER_HPP

#include <iostream>
#include <Eigen/Dense>
#include <iomanip>
#include <fstream>

namespace qst{

class Observer{

    Basis & basis_;
    Rbm & rbm_;
    ParallelTempering & pt_;

    int nv_;
    int nh_;
    int D_;
    int npar_;
    int nchains_;

    double negative_log_likelihood_;
    double kl_divergence_;
    double overlap_;
    double best_overlap_;
    double best_nll_;
    double logZ_;
    double infiniteT_Z_;

    Eigen::VectorXd wf_;
   
public:

     
    Observer(Basis &basis,Rbm &rbm, ParallelTempering & pt):basis_(basis),rbm_(rbm),pt_(pt){ 
        
        std::cout<<"- Initializing observer module"<<std::endl;
        nv_ = rbm_.Nvisible();
        D_ = basis_.Dimension();
        npar_ = rbm_.Npar();
        nchains_ = pt.Nchains();  
        wf_.resize(D_);
        best_overlap_ = 0.0;
        best_nll_ = 1000.0;
        infiniteT_Z_=D_*std::pow(2,rbm_.Nhidden());
    }
  
    //Compute different estimators for the training performance
    void Scan(int i,Eigen::MatrixXd &nll_test){
        ExactPartitionFunction();
        KL(); 
        Overlap();
        NLL(nll_test);
        if (negative_log_likelihood_<best_nll_){ 
            best_overlap_ = overlap_;
            best_nll_ = negative_log_likelihood_;
        }
        PrintStats(i);
    }

    //Compute the partition function by exact enumeration 
    void ExactPartitionFunction() {
        double Z =0.0;
        for(int i=0;i<basis_.states_bin_.rows();i++){
            Z += exp(rbm_.LogVal(basis_.states_bin_.row(i)));
        }
        logZ_ = log(Z);
    }

    //Compute the overlap
    void Overlap(){
        overlap_ = 0.0;
        for(int i=0;i<D_;i++){
            overlap_ += wf_(i)*rbm_.psi(basis_.states_bin_.row(i))/sqrt(exp(logZ_));
        }
    }

    //Compute KL divergence exactly
    void KL(){
        kl_divergence_ = 0.0;
        for(int i=0;i<D_;i++){
            kl_divergence_ += wf_(i)*wf_(i)*log(wf_(i)*(wf_(i)));
            kl_divergence_ += wf_(i)*wf_(i)*(logZ_ - rbm_.LogVal(basis_.states_bin_.row(i)));
        }
    }
   
    //Compute the average negativ log-likelihood
    void NLL(Eigen::MatrixXd & samples) {
        negative_log_likelihood_=0.0;
        for (int k=0;k<samples.rows();k++){
            negative_log_likelihood_ += logZ_ - log(rbm_.psi(samples.row(k))*rbm_.psi(samples.row(k)));
        }
        negative_log_likelihood_ /= double(samples.rows());
    }

    //Compare the partition function obtained with different algorithms
    void ComparePartitionFunctions(int i){
        std::cout << "Epoch: " << i << "\t";
        ExactPartitionFunction();
        std::cout << "Exact logZ = " << std::setprecision(10) << logZ_ << "\t";
        logZ_ = pt_.ComputeLogPartitionFunction(infiniteT_Z_);
        std::cout << "PT logZ = " << std::setprecision(10) << logZ_<< "\t";
        std::cout<<std::endl;
    }

    //Print observer
    void PrintStats(int i){
        std::cout << "Epoch: " << i << "\t";     
        std::cout << "KL = " << std::setprecision(10) << kl_divergence_ << "\t";
        std::cout << "NLL = " << std::setprecision(10) << negative_log_likelihood_ << "\t";
        std::cout << "Overlap = " << std::setprecision(10) << overlap_<< "\t";//<< Fcheck_;
        std::cout << "Best overlap = " << std::setprecision(10) << best_overlap_;// << "\t" << Fcheck_;
        std::cout << std::endl;
    } 

    //Set the value of the target wavefunction
    void setWavefunction(Eigen::VectorXd & psi){
        for(int i=0;i<D_;i++){
            wf_(i) = psi(i);
        }
    }
};
}

#endif
