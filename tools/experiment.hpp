#ifndef TOOLS_EXPERIMENT_HPP
#define TOOLS_EXPERIMENT_HPP
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>

namespace tools{


//*****************************************************************************

//Class containing the parameters of the simulation
class Experiment{

public:
   
    //Number of sites
    int nsites_;
    //Number of bosons
    int M_;
    //Lattice Dimensions
    int d_;
    //On-site interaction strength
    double U_;
    //Hopping strength
    double t_;
    //Chemical potential
    double mu_;

    //Number of hidden units
    int nh_;
    // Width of normal distribution for initial weights
    double w_;
    //Order of contrastive divergence
    int cd_;
    //Number of chains
    int nc_;
    //Learning rate
    double lr_;
    //L2 normalization
    double l2_;
    //Batch size
    int bs_;
    //Training iterations
    int ep_;
    //Number of training samples 
    int Ns_;
    //Natural gradient regularization
    double lambda_;
    
    //Optimization algorithm
    std::string opt_;
    //Model name
    std::string model_;

    Experiment() {
        //Initialize parameter
        nsites_ = 2;
        M_ = 2;
        d_ = 1;
        U_ = 1.0;
        t_ = 1.0;
        mu_ = 0.0;

        nh_ = 2;
        w_ = 0.01;
        cd_ = 10;
        nc_ = 10;
        lr_ = 0.01;
        l2_ = 0.0;
        bs_ = 100;
        ep_=100000;
        Ns_ = 0;
        lambda_=0.0;
        opt_ = "sgd";
        model_="";
    }
    
    //Read parameters from the command line
    void ReadParameters(int argc,char** argv){
        std::string flag;
        
        flag = "-nsites";
        for(int i=2;i<argc;i++){
            if(flag==argv[i]) nsites_=atoi(argv[i+1]);
        }
        flag = "-d";
        for(int i=2;i<argc;i++){
            if(flag==argv[i]) d_=atoi(argv[i+1]);
        }
        flag = "-M";
        for(int i=2;i<argc;i++){
            if(flag==argv[i]) M_=atoi(argv[i+1]);
        }
        flag = "-U";
        for(int i=2;i<argc;i++){
            if(flag==argv[i]) U_=double(atof(argv[i+1]));
        }
        flag = "-nh";
        for(int i=2;i<argc;i++){
            if(flag==argv[i]) nh_=atoi(argv[i+1]);
        }
        flag = "-w";
        for(int i=2;i<argc;i++){
            if(flag==argv[i]) w_=double(atof(argv[i+1]));
        }
        flag = "-nc";
        for(int i=2;i<argc;i++){
            if(flag==argv[i]) nc_=atoi(argv[i+1]);
        }
        flag = "-cd";
        for(int i=2;i<argc;i++){
            if(flag==argv[i]) cd_=atoi(argv[i+1]);
        }
        flag = "-lr";
        for(int i=2;i<argc;i++){
            if(flag==argv[i]) lr_=double(atof(argv[i+1]));
        }
        flag = "-l2";
        for(int i=2;i<argc;i++){
            if(flag==argv[i]) l2_=double(atof(argv[i+1]));
        }
        flag = "-bs";
        for(int i=2;i<argc;i++){
            if(flag==argv[i]) bs_=atoi(argv[i+1]);
        }
        flag = "-Ns";
        for(int i=2;i<argc;i++){
            if(flag==argv[i]) Ns_=atoi(argv[i+1]);
        }
        flag = "-ep";
        for(int i=2;i<argc;i++){
            if(flag==argv[i]) ep_=atoi(argv[i+1]);
        }
        flag = "-lambda";
        for(int i=2;i<argc;i++){
            if(flag==argv[i]) lambda_=double(atof(argv[i+1]));
        }
        flag = "-opt";
        for(int i=2;i<argc;i++){
            if(flag==argv[i]) opt_=argv[i+1];
        }
    }
    
    //Print all the parameters on screen
    void PrintParameters(){

        std::cout << "Neural-Network Quantum State Tomography of ultra-cold bosons\n\n";
        std::cout << " Dimension: " << d_ << std::endl;
        std::cout << " Number of sites: " << nsites_ << std::endl;
        std::cout << " Number of bosons: " << M_<< std::endl;
        std::cout << " Hopping strength: " << t_ << std::endl;
        std::cout << " Interaction strength: " << U_ << std::endl;
        std::cout << std::endl;
        std::cout << "Network architecture: RBM" << std::endl;
        std::cout << " Number of hidden units: " << nh_<< std::endl;
        std::cout << " Initial distribution width: " << w_<< std::endl;
        std::cout << " Optimization: " << opt_<< std::endl;
        std::cout << " Number of chains: " << nc_<< std::endl;
        std::cout << " Steps of contrastive divergence: " << cd_<< std::endl;
        std::cout << " Learning rate: " << lr_<< std::endl;
        std::cout << " L2 regularization: " << l2_<< std::endl;
        std::cout << " Batch size: " << bs_<< std::endl;
        std::cout << " Number of training samples: " << Ns_<< std::endl;
        std::cout << " Number of training iterations " << ep_<< std::endl;
        std::cout << " Natural gradient regularization = " << lambda_<< std::endl;
        std::cout << std::endl;
    }
};
}




#endif
