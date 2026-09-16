#include <ibeamlab/datasets.h>
#include <iostream>
int main(int argc,char** argv){if(argc!=2)return 2;ibeamlab::datasets::DatasetReader reader(argv[1]);std::cout<<reader.readAll().size()<<" records\n";}
