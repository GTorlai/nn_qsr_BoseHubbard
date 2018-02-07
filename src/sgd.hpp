#ifndef QST_SGD_HPP
#define QST_SGD_HPP

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <random>

namespace qst{

class Sgd{

    double eta_;
    int npar_;
    double l2reg_;
    double dropout_p_;
    double momentum_;
    
    std::mt19937 rgen_;

public:

    Sgd(double eta,double momentum,double l2reg,double dropout_p):eta_(eta),l2reg_(l2reg),dropout_p_(dropout_p),momentum_(momentum){
        npar_=-1;
        //std::cout<<"- Initializing the optimizer as: SGD"<<std::endl;
    }
    
    void SetNpar(int npar){
        npar_=npar;
    }
    
    void Update(const Eigen::VectorXd & deltaP,Eigen::VectorXd & pars){
        assert(npar_>0);
        
        std::uniform_real_distribution<double> distribution(0,1);
        for(int i=0;i<npar_;i++){
            if(distribution(rgen_)>dropout_p_){
                pars(i)=(1.-momentum_)*pars(i) - (deltaP(i)+l2reg_*pars(i))*eta_;
            }
        }
    }
    
    void getUpdates(const Eigen::VectorXd & derLog, Eigen::VectorXd & deltaP){
      
        deltaP = derLog;
    }
    
    void Reset(){
        
    }
};


}

#endif
