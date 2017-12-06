#ifndef TOOLS_OPERATIONS_HPP
#define TOOLS_OPERATIONS_HPP
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include "parameters.hpp"

namespace tools{

//Factorial of n
inline unsigned long long Factorial(int n){
    unsigned long long factorial=1;
    for (int i=1;i<n+1;i++) {
        factorial *= i;
    }
    return factorial;
    //return (unsigned long long)(std::tgamma(n+1)); 
}

//Convert binomial state into multinomial state
inline void BinomialToMultinomial(const int &nsites, const int &M, const Eigen::VectorXd &vbin, Eigen::VectorXd &vmul)
{
    vmul.setZero();
    for(auto n=0;n<nsites;++n){
        for(auto m=0;m<M+1;++m){
            if(int(vbin((M+1)*n+m))){
                vmul(n) = m;
            }
        }
    }
}

//Convert multinomial state into binomial state
inline void MultinomialToBinomial(const int &nsites, const int &M, const Eigen::VectorXd &vmul, Eigen::VectorXd &vbin)
{
    vbin.setZero();
    for(auto n=0;n<nsites;++n){
        vbin((M+1)*n+int(vmul(n))) = 1;
    }
}

////Convert multinomial state into binomial state
//inline void MultinomialToBinomial(const int &nsites, const int &M, const Eigen::VectorXd &vmul, Eigen::MatrixXf::RowXpr myrow)
//{
//    vbin.setZero();
//    for(auto n=0;n<nsites;++n){
//        vbin((M+1)*n+int(vmul(n))) = 1;
//    }
//}
//Logistic functions 
inline double logistic(double x){
    return 1./(1.+std::exp(-x));
}

inline void logistic(const Eigen::VectorXd & x,Eigen::VectorXd & y){
    for(int i=0;i<x.size();i++){
        y(i)=logistic(x(i));
    }
}

inline void logistic(const Eigen::MatrixXd & x,Eigen::MatrixXd & y){
    for(int i=0;i<x.rows();i++){
        for(int j=0;j<x.cols();j++){
            y(i,j)=logistic(x(i,j));
        }
    }
}

//Softmax functions
void softmax(const Eigen::VectorXd & x,Eigen::VectorXd & y){
    double norm=0.0;
    for (int i=0;i<x.rows();i++){
        norm += std::exp(x(i));
        y(i) = std::exp(x(i));
    }
    y /= norm;
}

void softmax(const Eigen::MatrixXd & x,Eigen::MatrixXd & y){
    Eigen::VectorXd tmp(x.rows());
    for(int i=0;i<x.rows();i++){
        softmax(x.row(i),tmp);
        y.row(i) = tmp;
    }
}
                
//Softplus functions
inline double ln1pexp(double x){
    if(x>30){
        return x;
    }
    //return std::log(1.+std::exp(x));
    return std::log1p(std::exp(x));
}

void ln1pexp(const Eigen::VectorXd & x,Eigen::VectorXd & y){
    for(int i=0;i<x.size();i++){
        y(i)=ln1pexp(x(i));
    }
}

}

#endif
