#ifndef QST_RBM_HPP
#define QST_RBM_HPP

#include <iostream>
#include <Eigen/Dense>
#include <random>
//#include <fstream>
//#include <vector>
//#include <iomanip>
//#include <bitset>

namespace qst{

class Rbm{

    //number of visible units
    int nv_;
    //number of hidden units
    int nh_;
    //Number of p
    int M_;
    // Number of sites
    int nsites_;
    //number of parameters
    int npar_;
    //weights
    Eigen::MatrixXd W_;
    //visible units bias
    Eigen::VectorXd b_;
    Eigen::MatrixXd bBatch_;
    //hidden units bias
    Eigen::VectorXd c_;
    Eigen::MatrixXd cBatch_;

    Eigen::VectorXd lnthetas_;
    //Eigen::VectorXd thetasnew_;
    //Eigen::VectorXd lnthetasnew_;
    
    int D_;
    //Random number generator 
    std::mt19937 rgen_;
    
public:
    Rbm(tools::Experiment &exp):nsites_(exp.nsites_),M_(exp.M_),nh_(exp.nh_),c_(exp.nh_),lnthetas_(exp.nh_){
       
        nv_ = (M_+1)*nsites_;
        npar_=nv_+nh_+nv_*nh_;

        std::random_device rd;
        //rgen_.seed(rd());
        rgen_.seed(13579);
        
        W_.setZero(nh_,nv_);
        b_.setZero(nv_);
        c_.setZero(nh_);
    
        std::cout<<"- Initializing rbm with "<<nv_<<" visible units";
        std::cout<<" and "<<nh_<<" hidden units"<<std::endl;
    }
 
    int Nvisible()const{
        return nv_;
    }
    int Nhidden()const{
        return nh_;
    }
    int Nsites()const{
        return nsites_;
    }
    int Npar()const{
        return npar_;
    }
    
    void InitRandomPars(int seed,double sigma){
        std::default_random_engine generator(seed);
        std::normal_distribution<double> distribution(0,sigma);
        
        for(int i=0;i<nh_;i++){
            for(int j=0;j<nv_;j++){
                W_(i,j)=distribution(generator);
            }
        }
        for(int j=0;j<nv_;j++){
            b_(j)=distribution(generator);
        }
        for(int i=0;i<nh_;i++){
            c_(i)=distribution(generator);
        }
    }
    
    //Conditional Probabilities 
    void ProbHiddenGivenVisible(const Eigen::MatrixXd &v,Eigen::MatrixXd &probs){
        tools::logistic(v*W_.transpose()+cBatch_,probs);
    }
    void ProbVisibleGivenHidden(const Eigen::MatrixXd & h,Eigen::MatrixXd & probs){
        Eigen::MatrixXd activations(h.rows(),nv_);
        activations = h*W_+bBatch_;
        Eigen::VectorXd one_site_activations(M_+1);
        Eigen::VectorXd one_site_probs(M_+1);
        for(int s=0;s<h.rows();s++){
            for (int n=0;n<nsites_;n++){
                //TODO OPTIMIZE
                //one_site_activations = activations.block(s,(M_+1)*n,s,(M_+1)*n+M_);
                for(int m=0;m<M_+1;m++){
                    one_site_activations(m) = activations(s,(M_+1)*n+m);
                }
                tools::softmax(one_site_activations,one_site_probs);
                for(int m=0;m<M_+1;m++){
                    probs(s,(M_+1)*n+m) = one_site_probs(m);
                }
            }
        }
    }

    //Compute derivative of log-probability
    Eigen::VectorXd DerLog(const Eigen::VectorXd & v){
        Eigen::VectorXd der(npar_);
        //der.setZero();
        int p=0;
       
        tools::logistic(W_*v+c_,lnthetas_);
        for(int i=0;i<nh_;i++){
            for(int j=0;j<nv_;j++){
                der(p)=lnthetas_(i)*v(j);
                p++;
            }
        }
        for(int j=0;j<nv_;j++){
            der(p)=v(j);
            p++;
        }
        
        for(int i=0;i<nh_;i++){
            der(p)=lnthetas_(i);
            p++;
        } 
        return der;
    }
    
    double amplitude(const Eigen::VectorXd & v){
        return exp(0.5*LogVal(v));
    }
    
    //Value of the logarithm of the RBM probability
    double LogVal(const Eigen::VectorXd & v){
        tools::ln1pexp(W_*v+c_,lnthetas_);
        return v.dot(b_)+lnthetas_.sum();
    }
    
    //Compute the partition function by exact enumeration 
    double ExactPartitionFunction(const Basis &basis) {
        double Z=0.0;
        Eigen::VectorXd v_bin(nsites_);
        for(int i=0;i<D_;i++){
            //mult_to_bin(basis_states_.row(i),v_bin);
            Z += exp(LogVal(basis.states_bin_.row(i)));//basis_states_.row(i)));
        }
        return Z;
    }
 
    // UTITILIES FUNCTIONS
    //Get RBM parameters
    Eigen::VectorXd GetParameters(){
        Eigen::VectorXd pars(npar_);
        int p=0;
        for(int i=0;i<nh_;i++){
            for(int j=0;j<nv_;j++){
                pars(p)=W_(i,j);
                p++;
            }
        }
        
        for(int j=0;j<nv_;j++){
            pars(p)=b_(j);
            p++;
        }
        
        for(int i=0;i<nh_;i++){
            pars(p)=c_(i);
            p++;
        }
        return pars;
    }
    
    //Set RBM parameters
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

    //Set the Biases matrices for batch sampling
    void SetBatchBiases(const int & bs) {
        bBatch_.resize(bs,nv_);
        cBatch_.resize(bs,nh_);
        for(int s=0;s<bs;s++) {
            bBatch_.row(s) = b_;
            cBatch_.row(s) = c_;
        }
    }
};
}

#endif
