#ifndef QST_BASIS_HPP
#define QST_BASIS_HPP

#include <iostream>
#include <Eigen/Core>

namespace qst{

class Basis{
    //Number of sites
    int nsites_;
    //Number of bosons
    int M_;
    //Number of binary variables
    int nv_;
    //Dimension of the Hilbert space
    unsigned long long D_;
    //Basis states in occupation number basis
    Eigen::MatrixXd basis_states_mul_;
    //Basis state in binary form
    Eigen::MatrixXd basis_states_bin_;
    
public:
    Basis(int nsites,int M):nsites_(nsites),M_(M){
        if (nsites_>10){
            std::cout<<"Dimension of the Hilbert space too large!"<<std::endl;
            exit(0);
        }
        else {
            D_ = tools::Factorial(M_+nsites_-1)/(tools::Factorial(M_)*tools::Factorial(nsites_-1));
            nv_ = (M_+1)*nsites_;
            basis_states_mul_.resize(D_,nsites_);
            basis_states_bin_.resize(D_,nv_);
        }   
    }
    inline int Nsites(){
        return nsites_;
    }
    inline int Nbosons(){
        return M_;
    }
    inline int Dim(){
        return D_;
    }

    //Generate the basis (multinomial and binomial)
    void GenerateBasis() {
        Eigen::VectorXd boson_number(nsites_);
        Eigen::VectorXd state(nv_);
        
        boson_number.setZero();
        basis_states_mul_.setZero();
        boson_number(0) = M_;
        basis_states_mul_.row(0) = boson_number;
        bool exit = false;
        int state_counter=0;
        while(exit==false){
            if (boson_number(0)>0){
                boson_number(0)--;
                boson_number(1)++;
                state_counter++;
                basis_states_mul_.row(state_counter) = boson_number;
            }
            else {
                int index=0;
                for (int i=0;i<nsites_;i++){
                    if (boson_number(i) != 0) {
                        index = i;
                        break;
                    }
                }
                if(index == nsites_-1){
                    exit = true;
                    break;
                }
                boson_number(0) = boson_number(index) -1;  
                boson_number(index) = 0;
                boson_number(index+1)++;
                state_counter++;
                basis_states_mul_.row(state_counter) = boson_number;
            }
        }
        for(int i=0;i<D_;i++){
            tools::MultinomialToBinomial(nsites_,M_,basis_states_mul_.row(i),state);
            basis_states_bin_.row(i) = state;
        }
    }
    
    //Print informations on the basis
    void PrintInfos(){
        std::cout<<"\nDimension of the physical Hilbert space: "<<D_*D_<<std::endl;
        std::cout<<std::endl;
        std::cout<<"Basis States:"<<std::endl<<std::endl;
        for(int i=0;i<D_;i++){
            std::cout<<"Multinomial =  "<<basis_states_mul_.row(i);
            std::cout<<"\tBinomial =  "<<basis_states_bin_.row(i)<<std::endl;
        }
    }
 
};
}

#endif
