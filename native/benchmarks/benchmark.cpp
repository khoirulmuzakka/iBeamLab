#include <ibeamlab/spectrum_processing.h>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>
int main(){std::vector<double> spectrum(16384,1),oldEdges(16385),newEdges(4097);std::iota(oldEdges.begin(),oldEdges.end(),0.0);for(std::size_t i=0;i<newEdges.size();++i)newEdges[i]=4.0*i;const auto start=std::chrono::steady_clock::now();std::size_t values=0;for(int i=0;i<1000;++i)values+=ibeamlab::spectrum::rebin(oldEdges,newEdges,spectrum).size();const auto time=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();std::cout<<values/time<<" output bins/s\n";}
