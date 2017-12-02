#ifndef TOOLS_NAMEFORMATTING_HPP
#define TOOLS_NAMEFORMATTING_HPP
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "parameters.hpp"
#include <boost/format.hpp>

namespace tools{

std::string TrainingDataName(Parameters & p){

    std::string fileName;
    fileName = "../data/bosehubbard";
    fileName += boost::str(boost::format("%d") %p.d_) + "d/";
    fileName += "datasets/bosehubbard";
    fileName += boost::str(boost::format("%d") %p.d_) + "d_N";
    fileName += boost::str(boost::format("%d") % p.nsites_);
    fileName += "_M";
    fileName += boost::str(boost::format("%d") % p.M_);
    fileName += "_U";
    fileName += boost::str(boost::format("%.2f") % p.U_); 
    fileName += "_S";
    fileName += boost::str(boost::format("%d") % p.Ns_);
    fileName += "_train.txt"; 
    //std::cout << fileName << std::endl;
    return fileName;
}

std::string TestingDataName(Parameters & p){

    std::string fileName;
    fileName = "../data/bosehubbard";
    fileName += boost::str(boost::format("%d") %p.d_) + "d/";
    fileName += "datasets/bosehubbard";
    fileName += boost::str(boost::format("%d") %p.d_) + "d_N";
    fileName += boost::str(boost::format("%d") % p.nsites_);
    fileName += "_M";
    fileName += boost::str(boost::format("%d") % p.M_);
    fileName += "_U";
    fileName += boost::str(boost::format("%.2f") % p.U_); 
    fileName += "_S";
    fileName += boost::str(boost::format("%d") % int(p.Ns_/10));
    fileName += "_test.txt"; 
    //std::cout << fileName << std::endl;
    return fileName;
}

std::string WavefunctionName(Parameters & p){
    std::string fileName;
    fileName = "../data/bosehubbard";
    fileName += boost::str(boost::format("%d") %p.d_) + "d/";
    fileName += "wavefunctions/wavefunction_bosehubbard";
    fileName += boost::str(boost::format("%d") %p.d_) + "d_N";
    fileName += boost::str(boost::format("%d") % p.nsites_);
    fileName += "_M";
    fileName += boost::str(boost::format("%d") % p.M_);
    fileName += "_U" + boost::str(boost::format("%.2f") % p.U_);
    fileName += ".txt";
    std::cout<<fileName<<std::endl;
    return fileName;
}

std::string NetworkName(Parameters & p){
    std::string fileName;

    fileName = "rbmState_bosehubbard";
    fileName += boost::str(boost::format("%d") %p.d_) + "d";
    fileName += "_N" + boost::str(boost::format("%d") % p.nsites_);
    fileName += "_M" + boost::str(boost::format("%d") % p.M_);
    fileName += "_nv" + boost::str(boost::format("%d") % p.nv_);
    fileName += "_nh" + boost::str(boost::format("%d") % p.nh_);
    fileName += "_" + p.opt_;
    fileName += "_lr";
    if (p.lr_ > 0.09) {
        fileName += boost::str(boost::format("%.1f") % p.lr_);
    }
    else if (p.lr_ > 0.009) {
        fileName += boost::str(boost::format("%.2f") % p.lr_);
    }
    else if (p.lr_ > 0.0009) {
        fileName += boost::str(boost::format("%.3f") % p.lr_);
    }
    else if (p.lr_ > 0.00009) {
        fileName += boost::str(boost::format("%.4f") % p.lr_);
    }
    else if (p.lr_ > 0.000009) {
        fileName += boost::str(boost::format("%.5f") % p.lr_);
    }
    fileName += "_bs" + boost::str(boost::format("%d") % p.bs_);
    fileName += "_w";
    if (p.w_ > 0.09) {
        fileName += boost::str(boost::format("%.1f") % p.w_);
    }
    else if (p.w_ > 0.009) {
        fileName += boost::str(boost::format("%.2f") % p.w_);
    }
    else if (p.w_ > 0.0009) {
        fileName += boost::str(boost::format("%.3f") % p.w_);
    }
 
    if (p.opt_ == "ngd"){
        fileName += "_lambda";
        if (p.lambda_ > 0.009) {
            fileName += boost::str(boost::format("%.2f") % p.lambda_);
        }
        else if (p.lambda_ > 0.0009) {
            fileName += boost::str(boost::format("%.3f") % p.lambda_);
        }
        else if (p.lambda_ > 0.00009) {
            fileName += boost::str(boost::format("%.4f") % p.lambda_);
        }
    }
    fileName += "_S";
    fileName += boost::str(boost::format("%d") % p.Ns_);
    fileName += "_U" + boost::str(boost::format("%.2f") % p.U_);
    return fileName;
}

std::string RbmWeightsName(Parameters & p){
    std::string fileName;
    fileName = "../data/bosehubbard";
    fileName += boost::str(boost::format("%d") %p.d_) + "d/";
    fileName += "weights/";
    fileName += NetworkName(p);
    fileName += "_weights.txt";
    return fileName;
}

std::string RbmObserverName(Parameters & p){
    std::string fileName;
    fileName = "../data/bosehubbard";
    fileName += boost::str(boost::format("%d") %p.d_) + "d/";
    fileName += "observers/";
    fileName += NetworkName(p);
    fileName += "_observer.txt";
    return fileName;
}


}

#endif
