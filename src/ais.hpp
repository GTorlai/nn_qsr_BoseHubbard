#ifndef QST_AIS_HPP
#define QST_AIS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <random>

namespace qst{

class AIS{

    //Number of chains
    int nreplicas_;
    int nchains_;
    int iterations_;
    int nv_;
    int nh_;
    int nh_br_; 
    int nsites_;
    int M_;
    double Z_br_;

    Eigen::MatrixXd W_;         //weights
    Eigen::VectorXd b_;         //visible fields
    Eigen::VectorXd c_;         //hidden fields
    Eigen::VectorXd lnthetas_;  //sigmoid estimator
 
    Eigen::VectorXd b_br_;

    //Eigen::MatrixXd v_;
    //Eigen::MatrixXd h_;
    //Eigen::MatrixXd probv_;
    //Eigen::MatrixXd probh_;
    Eigen::VectorXd v_;
    Eigen::VectorXd h_;
    Eigen::VectorXd probv_;
    Eigen::VectorXd probh_;

    Rbm & rbm_;

    Eigen::VectorXd beta_;
    //Random number generator 
    std::mt19937 rgen_;
    
public:
    //Constructor
    AIS(Rbm & rbm,int nsites,int M,int nchains,int nreplicas):rbm_(rbm){//,rbms_(nchains,rbm){
  
        std::cout<<"- Initializing the AIS"<<std::endl;
        nv_=rbm_.Nvisible();
        nh_=rbm_.Nhidden();
        nh_br_=0;//nh_;
        nsites_ = rbm_.Nsites();
        M_ = rbm_.MaxNbosons();
        nreplicas_ = nreplicas;
        nchains_ = nchains;
        //v_.resize(nchains_,nv_);
        //probv_.resize(nchains_,nv_);
        //h_.resize(nchains_,nh_);
        //probh_.resize(nchains_,nh_);
        v_.resize(nv_);
        probv_.resize(nv_);
        h_.resize(nh_);
        probh_.resize(nh_);
        W_.resize(nh_,nv_);
        b_.resize(nv_);
        c_.resize(nh_);
        lnthetas_.resize(nh_);
        beta_.resize(nreplicas_);
        b_br_.resize(rbm_.Nvisible());
        
        rgen_.seed(13212); 
        std::normal_distribution<double> distribution(0,0.1);
        for(int j=0;j<nv_;j++){
            b_br_(j) = distribution(rgen_);
        }

        for(int r=0;r<nreplicas_;r++){
            beta_(r) = double(r)/double(nreplicas_-1);
        }
        
        Z_br_ = std::pow(2,nh_br_);
        for (int j=0;j<nv_;j++){
            Z_br_ *= (1.0+std::exp(b_br_(j)));
        }

    }

    double GetProbability(int k){
        double pA,pB;
        pA = std::exp((1-beta_(k))*b_br_.dot(v_));
        //pA *= std::pow(2,nh_br_); 
        pB = std::exp(beta_(k)*b_.dot(v_));
        tools::ln1pexp(beta_(k)*(W_*v_+c_),lnthetas_);
        //std::cout<<pB<<std::endl;
        //std::cout<<lnthetas_.sum()<<std::endl;
        //std::cout<<beta_(k)*(W_*v_+c_).transpose()<<std::endl;
        pB *= std::exp(lnthetas_.sum());
        return pA*pB;

    }

    double ComputeLogPartitionFunction(){
        double W = 0.0;
        double Z = 0.0;
        LoadWeights();
        iterations_ = 100;
        for (int i=0;i<iterations_;i++){
            W = 1.0;
            SampleBaseRate();
            for(int r=1;r<nreplicas_;r++){
                for (int k=0;k<1;k++){
                    SampleStep(r-1);
                }
                W *= GetProbability(r)/GetProbability(r-1);
            }
            Z += W /double(iterations_);
        }
        double logZ = std::log(Z);
        //logZ += std::log(Z_br_);
        //logZ = std::log(Z_br_);
        return logZ;
    }


    void SampleBaseRate(){
        Eigen::VectorXd one_site_activations(M_+1);
        Eigen::VectorXd one_site_probs(M_+1);
        Eigen::VectorXd pr(M_+1);
        v_.setZero();
        //for(int s=0;s<nchains_;s++){
            for (int n=0;n<nsites_;n++){
                for(int m=0;m<M_+1;m++){
                    one_site_activations(m) = b_br_((M_+1)*n+m);//activations(s,(M_+1)*n+m);
                }
                tools::softmax(one_site_activations,one_site_probs);
                for(int m=0;m<M_+1;m++){
                    //probv_(s,(M_+1)*n+m) = one_site_probs(m);
                    probv_((M_+1)*n+m) = one_site_probs(m);
                }
            }
        //}
        //for(int s=0;s<v_.rows();s++){
            for(int n=0;n<nsites_;n++){
                for(int m=0; m<M_+1;m++){
                    //pr(m) = probv_(s,(M_+1)*n+m);
                    pr(m) = probv_((M_+1)*n+m);
                }
                std::discrete_distribution<int> distribution(pr.data(), pr.data()+pr.size());
                int site = distribution(rgen_);
                //v_(s,(M_+1)*n+site) = 1.0;
                v_((M_+1)*n+site) = 1.0;
            }
        //}
    }

    void SampleStep(int k){
        std::uniform_real_distribution<double> distribution(0,1);
        //Eigen::MatrixXd activations(h_.rows(),nv_);
        Eigen::VectorXd activations(nv_); 
        Eigen::VectorXd one_site_activations(M_+1);
        Eigen::VectorXd one_site_probs(M_+1);
        Eigen::VectorXd pr(M_+1);

        //tools::logistic(beta_(k)*((v_*W_.transpose()).rowwise() + c_.transpose()),probh_);
        tools::logistic(beta_(k)*((W_*v_)+c_),probh_);
        
        //for(int s=0;s<h_.rows();s++){
            for(int i=0;i<h_.cols();i++){
                //h_(s,i)=distribution(rgen_)<probh_(s,i);
                h_(i) = distribution(rgen_)<probh_(i);
            }
        //}
        //for(int s=0;s<h_.rows();s++){
        //    activations.row(s) = (1-beta_(k))*b_br_;
        //}
        activations = (1-beta_(k))*b_br_ + beta_(k)*(W_.transpose()*h_ + b_);
        //activations += beta_(k)*((h_*W_).rowwise() + b_.transpose());
        //for(int s=0;s<h_.rows();s++){
            for (int n=0;n<nsites_;n++){
                for(int m=0;m<M_+1;m++){
                    //one_site_activations(m) = activations(s,(M_+1)*n+m);
                    one_site_activations(m) = activations((M_+1)*n+m);
                }
                tools::softmax(one_site_activations,one_site_probs);
                for(int m=0;m<M_+1;m++){
                    //probv_(s,(M_+1)*n+m) = one_site_probs(m);
                    probv_((M_+1)*n+m) = one_site_probs(m);
                }
            }
        //}
        
        v_.setZero();
        //for(int s=0;s<v_.rows();s++){
            for(int n=0;n<nsites_;n++){
                for(int m=0; m<M_+1;m++){
                    //pr(m) = probv_(s,(M_+1)*n+m);
                    pr(m) = probv_((M_+1)*n+m);
                }
                std::discrete_distribution<int> distribution(pr.data(), pr.data()+pr.size());
                int site = distribution(rgen_);
                //v_(s,(M_+1)*n+site) = 1.0;
                v_((M_+1)*n+site) = 1.0;
            }
        //}
    }


    void LoadWeights() {
        auto pars = rbm_.GetParameters();
        SetParameters(pars);
    }

    //Set main RBM parameters
    void SetParameters(const Eigen::VectorXd & pars){
        int p=0;
        for(int i=0;i<nh_;i++){
            for(int j=0;j<nv_;j++){
                W_(i,j)=pars(p);
                p++;
            }
        }
        for(int j=0;j<nv_;j++){
            b_(j)=pars(p);
            p++;
        }
        for(int i=0;i<nh_;i++){
            c_(i)=pars(p);
            p++;
        }
    }
         
};
}

#endif
