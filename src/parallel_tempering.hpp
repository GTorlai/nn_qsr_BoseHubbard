#ifndef QST_PARALLELTEMPERING_HPP
#define QST_PARALLELTEMPERING_HPP
#include <iostream>
#include <Eigen/Dense>
#include <random>

namespace qst{

class ParallelTempering{
    
    Rbm & rbm_;                         //True rbm
    std::vector<Rbm> rbms_;             //Vector of Rbms
    std::vector<Gibbs> gibbs_;          //Vector of Gibbs samplers

    int nchains_;                       //Number of chains for the Gibbs sampler
    int nreplicas_;                     //Number of rebplicase at different temperature
    int iterations_;                    //Steps performed during PT

    Eigen::VectorXd beta_;              //Inverse temperatures
    Eigen::VectorXd swapAcceptance_;    //Acceptance statistics
    
    std::mt19937 rgen_;                 //Random number generator
    
public:
    //Constructor
    ParallelTempering(Rbm & rbm,int nsites,int M,int nchains,int nreplicas):rbm_(rbm),
                                                              rbms_(nreplicas,rbm){
  
        std::cout<<"- Initializing the sampler: Parallel Tempering"<<std::endl;
        nchains_ = nchains;
        nreplicas_ = nreplicas;
        beta_.resize(nreplicas_);
        swapAcceptance_.resize(nreplicas_);
        swapAcceptance_.setZero();

        auto par = rbm_.GetParameters();
        if (nreplicas_ == 1){
            std::cout << "Number of chains must be greater than 1" << std::endl;
            exit(0);
        }
        for(int r=0;r<nreplicas_;r++){
            beta_(r) = 1.0 - double(r)*1.0/double(nreplicas_-1);
            rbms_[r].SetParameters(par*beta_(r));
            gibbs_.push_back(Gibbs(rbms_[r],nsites,M,nchains_));
        }
        std::cout<<"- Initializing the sampler: DONE"<<std::endl;
    }
    
    int Nchains()const{
        return nchains_;
    }
    int Nreplicas()const{
        return nreplicas_;
    }
    void Reset(bool randomVal=false){
        auto par = rbm_.GetParameters();
        if (randomVal){
            for (int r=0;r<nreplicas_;r++){
                gibbs_[r].Reset(randomVal);
            }
        }
        for (int r=0;r<nreplicas_;r++){
            rbms_[r].SetParameters(par*beta_(r));
        }
    }

    Rbm & GetRbm(){
        return rbm_;
    }
   
    //Set the visible layer state
    void SetVisibleLayer(Eigen::MatrixXd v){
        gibbs_[0].SetVisibleLayer(v);
    }

    //Return the visible state 
    Eigen::MatrixXd VisibleStateRow(int s){
        return gibbs_[0].VisibleStateRow(s);
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

    //Perform k steps of Gibbs sampling
    void Sample(int steps){
        std::uniform_int_distribution<int> distribution(0,1);
        for (int i=0;i<steps;i++){
            for(int r=0;r<nreplicas_;r++){
                gibbs_[r].Sample(2);
            }
            for(int r=2;r<nreplicas_;r+=2){
                for(int c=0; c<nchains_;c++){ 
                    if(SwapProbability(r,r-1,c)>distribution(rgen_)){
                        Swap(r,r-1,c);
                        swapAcceptance_(r)++;
                        swapAcceptance_(r-1)++;
                    }
                }
            }
            for(int r=1;r<nreplicas_;r+=2){
                for(int c=0; c<nchains_;c++){ 
                    if(SwapProbability(r,r-1,c)>distribution(rgen_)){
                        Swap(r,r-1,c);
                        swapAcceptance_(r)++;
                        swapAcceptance_(r-1)++;
                    }
                }
            }
        }
        iterations_ += steps;
    }
  
    //Comput swap probability for 2 adiacent chains
    double SwapProbability(int replica1,int replica2,int chain){
        double E1 = rbms_[replica1].Energy(gibbs_[replica1].VisibleStateRow(chain),gibbs_[replica1].HiddenStateRow(chain));
        double E2 = rbms_[replica2].Energy(gibbs_[replica2].VisibleStateRow(chain),gibbs_[replica2].HiddenStateRow(chain));
        return std::min(1.0,std::exp((beta_(replica1)-beta_(replica2))*(E1-E2)));
    }

    //Swap two adiacent chains
    void Swap(int replica1,int replica2,int chain){
        auto v1 = gibbs_[replica1].VisibleStateRow(chain);
        auto h1 = gibbs_[replica2].HiddenStateRow(chain);

        gibbs_[replica1].SetVisibleLayerRow(chain,gibbs_[replica2].VisibleStateRow(chain));
        gibbs_[replica1].SetHiddenLayerRow(chain,gibbs_[replica2].HiddenStateRow(chain));
        gibbs_[replica2].SetVisibleLayerRow(chain,v1);
        gibbs_[replica2].SetHiddenLayerRow(chain,h1);
    }

    double ComputeLogPartitionFunction(double infiniteT_Z) {
        Reset();
        int eq = 10;
        double logZ = 0.0;
        int niter = 100;
        Eigen::VectorXd Z_ratio(nreplicas_-1);
        Z_ratio.setZero();
        Sample(eq);
        for(int i=0;i<niter;i++){
            Sample(1);
            for (int r=0;r<nreplicas_-1;r++){
                for(int c=0;c<nchains_;c++){
                    double E1 = rbms_[r].Energy(gibbs_[r+1].VisibleStateRow(c),gibbs_[r+1].HiddenStateRow(c));
                    double E2 = rbms_[r+1].Energy(gibbs_[r+1].VisibleStateRow(c),gibbs_[r+1].HiddenStateRow(c));
                    Z_ratio(r) += std::exp(-beta_(r)*E1+beta_(r+1)*E2)/double(niter*nchains_); 
                }
            }
        }
        for (int r=0;r<nreplicas_-1;r++){ 
            logZ += log(Z_ratio(r));
        }
        logZ += std::log(infiniteT_Z);
        return logZ;
    }
 
};
}

#endif
