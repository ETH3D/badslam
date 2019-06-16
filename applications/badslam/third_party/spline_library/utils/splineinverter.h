#pragma once

#include <vector>
#include <array>

#include <boost/math/tools/minima.hpp>

#include "../spline.h"
#include "splinesample_adaptor.h"

template<class InterpolationType, typename floating_t=float, size_t sampleDimension=2>
class SplineInverter
{
public:
    SplineInverter(const Spline<InterpolationType, floating_t> &spline, int samplesPerT = 10);

    floating_t findClosestT(const InterpolationType &queryPoint) const;

private: //methods
    SplineSamples<sampleDimension, floating_t> makeSplineSamples(int samplesPerT) const;

    static std::array<floating_t, sampleDimension> convertPoint(const InterpolationType &p);

private: //data
    const Spline<InterpolationType, floating_t> &spline;

    //distance in t between samples
    floating_t sampleStep;

    SplineSampleTree<sampleDimension, floating_t> sampleTree;
};

template<class InterpolationType, typename floating_t, size_t sampleDimension>
SplineInverter<InterpolationType, floating_t, sampleDimension>::SplineInverter(
        const Spline<InterpolationType, floating_t> &spline,
        int samplesPerT)
    :spline(spline), sampleStep(1.0 / samplesPerT), sampleTree(makeSplineSamples(samplesPerT))
{

}

template<class InterpolationType, typename floating_t, size_t sampleDimension>
SplineSamples<sampleDimension, floating_t> SplineInverter<InterpolationType, floating_t, sampleDimension>::makeSplineSamples(int samplesPerT) const
{
    SplineSamples<sampleDimension, floating_t> samples;
    floating_t maxT = spline.getMaxT();

    //find the number of segments we're going to use
    int numSegments = std::round(maxT * samplesPerT);

    for(int i = 0; i < numSegments; i++)
    {
        floating_t currentT = i * sampleStep;
        auto sampledPoint = convertPoint(spline.getPosition(currentT));
        samples.pts.emplace_back(sampledPoint, currentT);
    }

    //if the spline isn't a loop, add a sample for maxT
    if(!spline.isLooping())
    {
        auto sampledPoint = convertPoint(spline.getPosition(maxT));
        samples.pts.emplace_back(sampledPoint, maxT);
    }

    return samples;
}

template<class InterpolationType, typename floating_t, size_t sampleDimension>
floating_t SplineInverter<InterpolationType, floating_t, sampleDimension>::findClosestT(const InterpolationType &queryPoint) const
{
    auto convertedQueryPoint = convertPoint(queryPoint);
    floating_t closestSampleT = sampleTree.findClosestSample(convertedQueryPoint);

    //compute the first derivative of distance to spline at the sample point
    auto sampleResult = spline.getTangent(closestSampleT);
    InterpolationType sampleDisplacement = sampleResult.position - queryPoint;
    floating_t sampleDistanceSlope = InterpolationType::dotProduct(sampleDisplacement.normalized(), sampleResult.tangent);

    //if the spline is not a loop there are a few special cases to account for
    if(!spline.isLooping())
    {
        //if closest sample T is 0, we are on an end. so if the slope is positive, we have to just return the end
        if(closestSampleT == 0 && sampleDistanceSlope > 0)
            return 0;

        //if the closest sample T is max T we are on an end. so if the slope is negative, just return the end
        if(closestSampleT == spline.getMaxT() && sampleDistanceSlope < 0)
            return spline.getMaxT();
    }

    //step forwards or backwards in the spline until we find a point where the distance slope has flipped sign.
    //because "currentsample" is the closest point, the "next" sample's slope MUST have a different sign
    //otherwise that sample would be closer
    //note: this assumption is only true if the samples are close together

    //if sample distance slope is positive we want to move backwards in t, otherwise forwards
    floating_t a, b;
    if(sampleDistanceSlope > 0)
    {
        a = closestSampleT - sampleStep;
        b = closestSampleT;
    }
    else
    {
        a = closestSampleT;
        b = closestSampleT + sampleStep;
    }

    auto distanceFunction = [this, queryPoint](floating_t t) {
        return (spline.getPosition(t) - queryPoint).lengthSquared();
    };

    //we know that the actual closest T is now between a and b
    //use brent's method to find the actual closest point, using a and b as bounds
    auto result = boost::math::tools::brent_find_minima(distanceFunction, a, b, 16);
    return result.first;
}

template<class InterpolationType, typename floating_t, size_t sampleDimension>
std::array<floating_t, sampleDimension> SplineInverter<InterpolationType, floating_t, sampleDimension>::convertPoint(const InterpolationType &p)
{
    std::array<floating_t, sampleDimension> result;
    for(size_t i = 0; i < sampleDimension; i++) {
        result[i] = p[i];
    }
    return result;
}
