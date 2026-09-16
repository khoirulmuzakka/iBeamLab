#include <ibeamlab/data_generator.h>
#include <iostream>

int main(int argc,char** argv){
    if(argc<2||argc>3){std::cerr<<"usage: ibeamlab-generate-dummy OUTPUT [SAMPLES]\n";return 2;}
    try{
        const std::size_t count=argc==3?std::stoull(argv[2]):100;
        using namespace ibeamlab;
        sample::SampleModel sample{{sample::Layer{1.0,0,0,0,{sample::Species{"Si",1.0,{}}}}}};
        sample::ExperimentalSetup setup{{sample::Detector{"RBS",sample::Beam{"He",2000,10},1,1e9,0,0,15}}};
        generation::GenerationConfig config{sample,setup,{{"RBS","RBS",""}},{{"energy",generation::BeamEnergy{"RBS"},1000,3000,std::nullopt,"keV"}}};
        auto simulator=std::make_shared<simulator::DummySimulator>(1024);
        generation::DataGenerator generator(std::move(config),std::move(simulator));
        const auto summary=generator.generate(argv[1],{.samples=count,.batchSize=64,.shardCount=1,.seed=42,.failurePolicy=generation::FailurePolicy::Stop},[](const auto& p){std::cout<<'\r'<<p.attempted<<'/'<<p.total<<std::flush;});
        std::cout<<"\naccepted "<<summary.accepted<<" samples\n";return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
