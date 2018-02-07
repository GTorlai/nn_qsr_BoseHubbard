#ifndef QST_GIBBS_HPP
#define QST_GIBBS_HPP
#include <iostream>
#include <Eigen/Dense>
#include <random>

namespace qst{

class Gibbs{

    Rbm & rbm_;                 //Rbm class
    
    int nsites_;                //Number of sites 
    int M_;                     //Maximum site occupancy
    int nv_;                    //Number of visible units
    int nh_;                    //Number of hidden units
    int nchains_;               //Number of sampling chains
    
    Eigen::MatrixXd v_;         //Visible states
    Eigen::MatrixXd h_;         //Hidden states
    Eigen::MatrixXd probv_;     //Visible probabilities
    Eigen::MatrixXd probh_;     //Hidden probabilities
    std::mt19937 rgen_;         //Random number generator
    
public:
    //Constructor
    Gibbs(Rbm & rbm,int nsites,int M,int nchains):rbm_(rbm),
                                                  nsites_(nsites),
                                                  M_(M),
                                                  nchains_(nchains){
        //std::cout<<"- Initializing the sampler: Gibbs"<<std::endl;
        nv_ = rbm_.Nvisible();
        nh_ = rbm_.Nhidden();
        v_.resize(nchains_,nv_);
        h_.resize(nchains_,nh_);
        probv_.resize(nchains_,nv_);
        probh_.resize(nchains_,nh_);
        RandomVals(v_);
    }
  
    //Reset the sampler
    void Reset(bool randomVal=false){
        if (randomVal){
            RandomVals(v_);
        }
    }
   
    //Private member access functions
    inline int Nchains(){
        return nchains_;
    }
    Rbm & GetRbm(){
        return rbm_;
    }
    inline Eigen::VectorXd VisibleStateRow(int s){
        return v_.row(s);
    }
    inline Eigen::VectorXd HiddenStateRow(int s){
        return h_.row(s);
    }

    //Set the visible layer state
    inline void SetVisibleLayer(Eigen::MatrixXd v){
        v_=v;
    }
    //Set the visible layer state
    inline void SetHiddenLayer(Eigen::MatrixXd h){
        h_=h;
    }
    //Set the visible layer state
    inline void SetVisibleLayerRow(int s,Eigen::VectorXd v){
        v_.row(s)=v;
    }
    //Set the visible layer state
    inline void SetHiddenLayerRow(int s,Eigen::VectorXd h){
        h_.row(s)=h;
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
        v.setZero();
        Eigen::VectorXd pr(M_+1);
        for(int s=0;s<v.rows();s++){
            for(int n=0;n<nsites_;n++){
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
};
}

#endif
