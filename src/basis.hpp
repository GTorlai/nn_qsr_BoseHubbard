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
    //Maximum bosons occupaty at any site
    int M_max_;
    //Number of binary variables
    int nv_;
    //Dimension of the Hilbert space
    unsigned long long D_;

public:
    
    //Basis states in occupation number basis
    Eigen::MatrixXd states_mul_;
    //Basis state in binary form
    Eigen::MatrixXd states_bin_;
    
    Basis(int nsites,int M,int M_max):nsites_(nsites),M_(M),M_max_(M_max){
        GetDimension();
        nv_ = (M_max+1)*nsites_;
        states_mul_.resize(D_,nsites_);
        states_bin_.resize(D_,nv_);
        GenerateBasis();
    }
    
    inline int Nsites(){
        return nsites_;
    }
    inline int Nbosons(){
        return M_;
    }
    inline int Dimension(){
        return D_;
    }

    void GetDimension(){
        //Full basis
        if (M_ == M_max_){
            D_ = tools::Factorial(M_+nsites_-1)/(tools::Factorial(M_)*tools::Factorial(nsites_-1));
            //std::cout << "Dimension = " << D_ << std::endl;
        }
        //Restriced basis
        else {
            D_= GetNumCoefficient(M_max_,M_,nsites_);
            //std::cout << "Dimension = " << D_ << std::endl;
        }
    }

    //Generate the basis (multinomial and binomial)
    void GenerateBasis() {
        //Full basis
        if (M_ == M_max_){
            FullBasis();
        }
        //Restriced basis
        else {
            RestrictedBasis();
        }
     
    }

    //Get coefficient for restricted basis
    int GetNumCoefficient(int M_max, int M, int nsites){
        int C = 0;
        if ((M==0) && (nsites == 0)){
            return 1;
        }
        if (M<0) {
            return 0;
        }
        if (M_max*nsites < M){
            return 0;
        }
        else { 
            for(int L=M-M_max;L<M+1;L++){
                C += GetNumCoefficient(M_max,L,nsites-1);
            }
        }
        return C;
    }

    //Generate the basis (multinomial and binomial)
    void FullBasis() {
        Eigen::VectorXd boson_number(nsites_);
        Eigen::VectorXd state(nv_);
        
        boson_number.setZero();
        states_mul_.setZero();
        boson_number(0) = M_;
        states_mul_.row(0) = boson_number;
        bool exit = false;
        int state_counter=0;
        while(exit==false){
            if (boson_number(0)>0){
                boson_number(0)--;
                boson_number(1)++;
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
            }
            state_counter++;
            states_mul_.row(state_counter) = boson_number;
        }
        for(int i=0;i<D_;i++){
            tools::MultinomialToBinomial(nsites_,M_,states_mul_.row(i),state);
            states_bin_.row(i) = state;
        }
    }

    //Generate the basis (multinomial and binomial)
    void RestrictedBasis() {
        Eigen::VectorXd boson_number(nsites_);
        Eigen::VectorXd state(nv_);
        
        boson_number.setZero();
        states_mul_.setZero();
        int M_div_Mmax = std::floor(double(M_)/double(M_max_));
        for (int i=0;i<M_div_Mmax;i++){
            boson_number(i) = M_max_;
        }
        boson_number(M_div_Mmax) = M_-M_max_*M_div_Mmax;
        states_mul_.row(0) = boson_number;
        bool exit = false;
        int state_counter=0;
        int delta;
        for(int s=1;s<D_;s++){
            if (boson_number(0)>0){
                delta = 0;
                if (boson_number(0) < M_max_){
                    delta = M_max_-boson_number(0);
                    boson_number(0) = M_max_;
                }
                int index;
                for (int i=1;i<nsites_;i++){
                    if((boson_number(i)<M_max_)){
                        index = i;
                        break;
                    }
                }
                boson_number(index)++;
                boson_number(index-1) -= 1 + delta;
            }
            else {
                int index_j;
                int index_k;
                for (int j=1;j<nsites_;j++){
                    if((boson_number(j)>0)){
                        index_j = j;
                        break;
                    }
                }
                for (int i=index_j+1;i<nsites_;i++){
                    if((boson_number(i)<M_max_)){
                        index_k=i;
                        break;
                    }
                }
                boson_number(index_k)++;
                boson_number(index_k-index_j-1) = boson_number(index_j)-1;
                for (int l=0;l<index_k-index_j-1;l++){
                    boson_number(l) = M_max_;
                }
                for(int l=index_k-index_j;l<index_k;l++){
                    boson_number(l)= 0;
                }
            }
            state_counter++;
            states_mul_.row(state_counter) = boson_number;
        }
        for(int i=0;i<D_;i++){
            tools::MultinomialToBinomial(nsites_,M_max_,states_mul_.row(i),state);
            states_bin_.row(i) = state;
        }
    }
    
    //Print informations on the basis
    void PrintInfos(){
        std::cout<<"\nDimension of the physical Hilbert space: "<<D_*D_<<std::endl;
        std::cout<<std::endl;
        std::cout<<"Basis States:"<<std::endl<<std::endl;
        for(int i=0;i<D_;i++){
            std::cout<<"Multinomial =  "<<states_mul_.row(i);
            std::cout<<"\tBinomial =  "<<states_bin_.row(i)<<std::endl;
        }
    }
};
}

#endif
