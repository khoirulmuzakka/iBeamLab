#include <ibeamlab/fitting/fit.h>
#include <ibeamlab/profile/discretization.h>
#include <iostream>
#include <stdexcept>
using namespace ibeamlab;
void check(bool value) {
    if (!value)
        throw std::runtime_error("profile regression failed");
}
int main() {
    try {
        profile::Function f{profile::FunctionKind::Polynomial, {1, -1}};
        check(std::abs(f.evaluate(25, 0, 100) - .75) < 1e-12);
        check(f.evaluate(-1, 0, 100) == 0);
        profile::Function erf{profile::FunctionKind::ErfPolynomial, {.5, .2, .4, .1, .3}};
        check(std::abs(erf.evaluate(.4, 0, 1) - .56) < 1e-12);
        profile::Model m;
        core::Layer reference;
        reference.thickness = 100;
        reference.species = {{"Si", .5, {}}, {"O", .5, {}}};
        m.layers.push_back({reference,
                            {f, {profile::FunctionKind::Polynomial, {0, 1}}},
                            2,
                            profile::Spacing::Linear});
        profile::Discretizer grid(m);
        core::Sample sample;
        grid.apply(m, sample);
        check(sample.layers.size() == 2 && sample.layers[0].thickness == 50);
        check(std::abs(sample.layers[0].species[0].concentration - .75) < 1e-12);
        check(std::abs(sample.layers[1].species[0].concentration - .25) < 1e-12);
        auto ptr = sample.layers.data();
        auto species = sample.layers[0].species.data();
        m.layers[0].reference.thickness = 200;
        grid.apply(m, sample);
        check(sample.layers.data() == ptr && sample.layers[0].species.data() == species &&
              sample.layers[0].thickness == 100);
        m.layers[0].spacing = profile::Spacing::Logarithmic;
        bool rejected = false;
        try {
            grid.apply(m, sample);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        check(rejected);
        profile::Discretizer logarithmic(m);
        logarithmic.apply(m, sample);
        check(std::abs(sample.layers[0].thickness + sample.layers[1].thickness - 200) < 1e-12);
        check(fitting::loss_value(fitting::Loss::Neyman, 2, 4) == 2);
        check(fitting::loss_value(fitting::Loss::DataChiSquare, 2, 4) == 4. / 3);
        check(fitting::loss_value(fitting::Loss::Mse, 2, 4) == 4);
        std::cout << "profile evaluation, discretization, storage reuse and loss checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
