#include <ibeamlab/simnra_simulator.h>
#include <cassert>
int main(int argc,char** argv){if(argc!=2)return 2;using namespace ibeamlab;simulator::SimnraSimulatorConfig config{{{"RBS",argv[1]}},1};simulator::SimnraSimulator engine(config);sample::SampleModel sample{{sample::Layer{100,0,0,0,{sample::Species{"Si",1,{}}}}}};sample::ExperimentalSetup setup{{sample::Detector{"RBS",sample::Beam{"He",2000,10},1,1e9,0,0,15}}};const auto result=engine.simulateBatch({{sample,setup}});assert(result.size()==1&&!result[0].failure&&!result[0].spectra.empty());engine.close();engine.close();}
