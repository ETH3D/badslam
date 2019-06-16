#pragma once

#include <cmath>
#include <array>

class SplineLibraryCalculus {
private:
    SplineLibraryCalculus() = default;

public:
    //use the gauss-legendre quadrature algorithm to numerically integrate f from a to b
    //as of this writing, hardcoded to use 13 points
    template<class IntegrandType, class Function, typename floating_t>
    inline static IntegrandType gaussLegendreQuadratureIntegral(Function f, floating_t a, floating_t b)
    {
        const size_t NUM_POINTS = 13;

        //these are precomputed :( It would be cool to compute these at compile time, but apparently
        //it's not easy to compute the points/weights just given the number of points.
        //it involves computing every root of a polynomial. which can obviously be done, but not in a reasonable amount of code
        std::array<floating_t, NUM_POINTS> quadraturePoints = {
            floating_t( 0.0000000000000000),
            floating_t(-0.2304583159551348),
            floating_t( 0.2304583159551348),
            floating_t(-0.4484927510364469),
            floating_t( 0.4484927510364469),
            floating_t(-0.6423493394403402),
            floating_t( 0.6423493394403402),
            floating_t(-0.8015780907333099),
            floating_t( 0.8015780907333099),
            floating_t(-0.9175983992229779),
            floating_t( 0.9175983992229779),
            floating_t(-0.9841830547185881),
            floating_t( 0.9841830547185881)
        };

        std::array<floating_t, NUM_POINTS> quadratureWeights = {
            floating_t(0.2325515532308739),
            floating_t(0.2262831802628972),
            floating_t(0.2262831802628972),
            floating_t(0.2078160475368885),
            floating_t(0.2078160475368885),
            floating_t(0.1781459807619457),
            floating_t(0.1781459807619457),
            floating_t(0.1388735102197872),
            floating_t(0.1388735102197872),
            floating_t(0.0921214998377285),
            floating_t(0.0921214998377285),
            floating_t(0.0404840047653159),
            floating_t(0.0404840047653159)
        };

        floating_t halfDiff = (b - a) / 2;
        floating_t halfSum = (a + b) / 2;

        IntegrandType sum{};
        for(size_t i = 0; i < NUM_POINTS; i++)
        {
            sum += quadratureWeights[i] * f(halfDiff * quadraturePoints[i] + halfSum);
        }
        return halfDiff * sum;
    }
};
