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
    tools::Parameters & p_;

    int nv_;
    int nh_;
    int nsites_;
    int M_;
    int D_;
    int npar_;
    
    Eigen::VectorXd wf_;

public:

    double negative_log_likelihood_;
    double kl_divergence_;
    double overlap_;
     
    Observer(Basis &basis,Rbm &rbm,tools::Parameters &p):
             basis_(basis),rbm_(rbm),p_(p){ 
        
        std::cout<<"- Initializing observer module"<<std::endl;
        nv_ = (p_.M_+1)*p_.nsites_;
        D_ = basis_.Dimension();
        npar_ = p_.nv_*p_.nh_+p_.nv_+p_.nh_;
    }

   
    //Compute the overlap
    void Overlap(const double &Z){
        overlap_ = 0.0;
        for(int i=0;i<D_;i++){
            overlap_ += wf_(i)*rbm_.amplitude(basis_.states_bin_.row(i))/sqrt(Z);
        }
    }

    //Compute KL divergence exactly
    void ExactKL(const double &Z){
        kl_divergence_ = 0.0;
        for(int i=0;i<D_;i++){
            kl_divergence_ += wf_(i)*wf_(i)*log(wf_(i)*(wf_(i)));
            kl_divergence_ += wf_(i)*wf_(i)*(log(Z) - rbm_.LogVal(basis_.states_bin_.row(i)));
        }
    }
   
    //Compute the average negativ log-likelihood
    void NLL(Eigen::MatrixXd & samples,const double &Z) {
        negative_log_likelihood_=0.0;
        for (int k=0;k<samples.rows();k++){
            negative_log_likelihood_ -=log(rbm_.amplitude(samples.row(k))*rbm_.amplitude(samples.row(k))/Z);
        }
        negative_log_likelihood_ /= double(samples.rows());
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
