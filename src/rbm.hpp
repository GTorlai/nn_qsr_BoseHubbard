#ifndef QST_RBM_HPP
#define QST_RBM_HPP
#include <iostream>
#include <Eigen/Dense>
#include <random>

namespace qst{

class Rbm{

    int nv_;                    //number of visible units
    int nh_;                    //number of hidden units
    int M_;                     //maximum site occupancy
    int nsites_;                //number of sites
    int npar_;                  //number of parameters
    Eigen::MatrixXd W_;         //weights
    Eigen::VectorXd b_;         //visible fields
    Eigen::VectorXd c_;         //hidden fields
    Eigen::VectorXd lnthetas_;  //sigmoid estimator
    std::mt19937 rgen_;         //Random number generator
    
public:
    Rbm(tools::Parameters &par):nsites_(par.nsites_),
                                M_(par.M_max_),
                                nh_(par.nh_){
        //std::cout<<"- Initializing rbm with "<<nv_<<" visible units";
        //std::cout<<" and "<<nh_<<" hidden units"<<std::endl;
        nv_ = (M_+1)*nsites_;
        npar_=nv_+nh_+nv_*nh_;
        //std::random_device rd;
        //rgen_.seed(rd());
        rgen_.seed(13579);
        W_.resize(nh_,nv_);
        b_.resize(nv_);
        c_.resize(nh_);
        lnthetas_.resize(nh_);
    }

    //Private members access functions
    inline int Nvisible()const{
        return nv_;
    }
    inline int Nhidden()const{
        return nh_;
    }
    inline int Nsites()const{
        return nsites_;
    }
    inline int MaxNbosons()const{
        return M_;
    }
    inline int Npar()const{
        return npar_;
    }
   
    //Initialize the network parameters
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

    //Compute derivative of log-probability
    Eigen::VectorXd DerLog(const Eigen::VectorXd & v){
        Eigen::VectorXd der(npar_);
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
   
    //Return the wavefunction's amplitude
    inline double psi(const Eigen::VectorXd & v){
        return exp(0.5*LogVal(v));
    }
    
    //Value of the logarithm of the RBM probability
    inline double LogVal(const Eigen::VectorXd & v){
        tools::ln1pexp(W_*v+c_,lnthetas_);
        return v.dot(b_)+lnthetas_.sum();
    }
    
    //RBM Energy
    inline double Energy(const Eigen::VectorXd & v, const Eigen::VectorXd & h){
        return -h.dot(W_*v)-b_.dot(v)-c_.dot(h);
    }

    //Conditional Probabilities 
    void ProbHiddenGivenVisible(const Eigen::MatrixXd &v,Eigen::MatrixXd &probs){
        tools::logistic((v*W_.transpose()).rowwise() + c_.transpose(),probs);
    }
    void ProbVisibleGivenHidden(const Eigen::MatrixXd & h,Eigen::MatrixXd & probs){
        Eigen::MatrixXd activations(h.rows(),nv_);
        //activations = h*W_+bBatch_;
        activations = (h*W_).rowwise() + b_.transpose();
        Eigen::VectorXd one_site_activations(M_+1);
        Eigen::VectorXd one_site_probs(M_+1);
        for(int s=0;s<h.rows();s++){
            for (int n=0;n<nsites_;n++){
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
   
    //Print the weights
    void PrintWeights(){
        std::cout<<W_<<std::endl;
        std::cout<<b_.transpose()<<std::endl;
        std::cout<<c_.transpose()<<std::endl;
        std::cout<<std::endl<<std::endl;
    }

 };
}

#endif
