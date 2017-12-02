#ifndef QST_GIBBS_HPP
#define QST_GIBBS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <random>
//#include <vector>
//#include <iomanip>
//#include <fstream>
//#include <bitset>

namespace qst{

class Gibbs{

    Rbm & rbm_;
    //Number of sites
    int nsites_;
    //Number of bosons
    int M_;
    //Number of binary variables
    int nv_;
    //Number of hiddne units
    int nh_;
    //Number of chains
    int nchains_;

    Eigen::MatrixXd v_;
    Eigen::MatrixXd h_;
    Eigen::MatrixXd probv_;
    Eigen::MatrixXd probh_;

    //Random number generator 
    std::mt19937 rgen_;
    
public:
    //Constructor
    Gibbs(Rbm & rbm,tools::Parameters &par):rbm_(rbm),nsites_(par.nsites_),M_(par.M_),nh_(par.nh_){
  
        std::cout<<"- Initializing the sampler: Gibbs"<<std::endl;
        nv_ = (M_+1)*nsites_;
        nchains_ = par.nc_;
        std::uniform_int_distribution<int> distribution(0,1);
        
        v_.resize(nchains_,nv_);
        h_.resize(nchains_,nh_);
        probv_.resize(nchains_,nv_);
        probh_.resize(nchains_,nh_);
        
        RandomVals(v_);
        //PrintVisible();
    }
    
    Rbm & Rbm(){
        return rbm_;
    }
    
    Eigen::VectorXd VisibleStateRow(int s){
        return v_.row(s);
    }

    //Set a state to a random binary value
    void RandomVals(Eigen::MatrixXd & hv){
        std::uniform_int_distribution<int> distribution(0,1);
        for(int i=0;i<hv.rows();i++){
            for(int j=0;j<hv.cols();j++){
                hv(i,j)=distribution(rgen_);
            }
        }
    }
    
    //Set the visible layer state
    void SetVisibleLayer(Eigen::MatrixXd v){
        v_=v;
    }
    //Set the hidden layer
    void SetHiddenLayer(Eigen::MatrixXd h){
        h_=h;
    }

    //Sample the hidden layer 
    void SampleHidden(Eigen::MatrixXd & h,const Eigen::MatrixXd & probs){
        std::uniform_real_distribution<double> distribution(0,1);
        for(int s=0;s<h.rows();s++){
            for(int i=0;i<h.cols();i++){
                h(s,i)=distribution(rgen_)<probs(s,i);
            }
        }
    }

    //Samples the visible layer
    void SampleVisible(Eigen::MatrixXd & v,const Eigen::MatrixXd & probs){
        //v.setZero(int(v.rows()),nv_);
        v.setZero();
        Eigen::VectorXd pr(M_+1);
        for(int s=0;s<v.rows();s++){
            for(int n=0;n<nsites_;n++){
                //TODO OPTIMIZE
                //pr = probs.block(s,(M_+1)*n,s,(M_+1)*n+M_);
                for(int m=0; m<M_+1;m++){
                    pr(m) = probs(s,(M_+1)*n+m);
                }
                std::discrete_distribution<int> distribution(pr.data(), pr.data()+pr.size());
                int site = distribution(rgen_);
                v(s,(M_+1)*n+site) = 1.0;
            }
        }
    }

    //Perform k steps of Gibbs sampling
    void Sample(int steps){
        for(int k=0;k<steps;k++){
            rbm_.ProbHiddenGivenVisible(v_,probh_);
            SampleHidden(h_,probh_);
            rbm_.ProbVisibleGivenHidden(h_,probv_);
            SampleVisible(v_,probv_);
        }
    }
    void PrintVisible(){
        std::cout<< v_ << std::endl;
    }

};
}

#endif
