#ifndef TOOLS_DATA_HPP
#define TOOLS_DATA_HPP
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include "parameters.hpp"

namespace tools{

void LoadTrainingData(Parameters & p,Eigen::MatrixXd & trainSamples) 
{
    int train_size = int(0.9*p.Ns_);
    trainSamples.resize(train_size,p.Ns_);
    std::ifstream samplesFile(tools::TrainingDataName(p));
    
    for (int n=0; n<train_size; n++) {
        for (int j=0; j<p.nsites_; j++) {
            samplesFile >> trainSamples(n,j);
            //cout << trainBases[n][j] << " ";
        }
        //cout << endl;
    }
}

void LoadTestingData(Parameters & p,Eigen::MatrixXd & trainSamples) 
{
    int train_size = int(0.9*p.Ns_);
    trainSamples.resize(train_size,p.Ns_);
    std::ifstream samplesFile(tools::TestingDataName(p));
    
    for (int n=0; n<train_size; n++) {
        for (int j=0; j<p.nsites_; j++) {
            samplesFile >> trainSamples(n,j);
            //cout << trainBases[n][j] << " ";
        }
        //cout << endl;
    }
}
void LoadWavefunction(Parameters & p,Eigen::VectorXd & wf,int D){
    
    std::ifstream wfFile(tools::WavefunctionName(p));
    wf.resize(D);
    for(int i=0;i<D;i++){
        wfFile>>wf(i);
    }
    wfFile.close();
}

}

#endif
