#pragma once

#include <cassert>

#include "../spline.h"

template<class InterpolationType, typename floating_t>
class UniformCubicBSplineCommon
{
public:

    inline UniformCubicBSplineCommon(void) = default;
    inline UniformCubicBSplineCommon(std::vector<InterpolationType> points)
        :points(std::move(points))
    {}

    inline size_t segmentCount(void) const
    {
        return points.size() - 3;
    }

    inline size_t segmentForT(floating_t t) const
    {
        if(t < 0)
            return 0;

        size_t segmentIndex = size_t(t);
        if(segmentIndex > segmentCount() - 1)
            return segmentCount() - 1;
        else
            return segmentIndex;
    }

    inline floating_t segmentT(size_t segmentIndex) const
    {
        return segmentIndex;
    }

    inline InterpolationType getPosition(floating_t globalT) const
    {
        size_t segmentIndex = segmentForT(globalT);
        floating_t localT = globalT - segmentIndex;

        return computePosition(segmentIndex, localT);
    }

    inline typename Spline<InterpolationType,floating_t>::InterpolatedPT getTangent(floating_t globalT) const
    {
        size_t segmentIndex = segmentForT(globalT);
        floating_t localT = globalT - segmentIndex;

        return typename Spline<InterpolationType,floating_t>::InterpolatedPT(
                    computePosition(segmentIndex, localT),
                    computeTangent(segmentIndex, localT)
                    );
    }

    inline typename Spline<InterpolationType,floating_t>::InterpolatedPTC getCurvature(floating_t globalT) const
    {
        size_t segmentIndex = segmentForT(globalT);
        floating_t localT = globalT - segmentIndex;

        return typename Spline<InterpolationType,floating_t>::InterpolatedPTC(
                    computePosition(segmentIndex, localT),
                    computeTangent(segmentIndex, localT),
                    computeCurvature(segmentIndex, localT)
                    );
    }

    inline typename Spline<InterpolationType,floating_t>::InterpolatedPTCW getWiggle(floating_t globalT) const
    {
        size_t segmentIndex = segmentForT(globalT);
        floating_t localT = globalT - segmentIndex;

        return typename Spline<InterpolationType,floating_t>::InterpolatedPTCW(
                    computePosition(segmentIndex, localT),
                    computeTangent(segmentIndex, localT),
                    computeCurvature(segmentIndex, localT),
                    computeWiggle(segmentIndex)
                    );
    }

    inline floating_t segmentLength(size_t index, floating_t a, floating_t b) const
    {
        auto segmentFunction = [this, index](floating_t t) -> floating_t {
            auto tangent = computeTangent(index, t);
            return tangent.length();
        };

        floating_t localA = a - index;
        floating_t localB = b - index;

        return SplineLibraryCalculus::gaussLegendreQuadratureIntegral<floating_t>(segmentFunction, localA, localB);
    }

private: //methods
    inline InterpolationType computePosition(size_t index, floating_t t) const
    {
        return (
                    points[index] * ((1 - t) * (1 - t) * (1 - t)) +
                    points[index + 1] * (t * t * 3 * (t - 2) + 4) +
                    points[index + 2] * (t * (t * (-3 * t + 3) + 3) + 1) +
                    points[index + 3] * (t * t * t)
                ) / floating_t(6);
    }

    inline InterpolationType computeTangent(size_t index, floating_t t) const
    {
        return (
                    points[index] * (-(1 - t) * (1 - t)) +
                    points[index + 1] * (t * (3 * t - 4)) +
                    points[index + 2] * ((3 * t + 1) * (1 - t)) +
                    points[index + 3] * (t * t)
                ) / floating_t(2);
    }

    inline InterpolationType computeCurvature(size_t index, floating_t t) const
    {
        return (
                    points[index] * (1 - t) +
                    points[index + 1] * (3 * t - 2) +
                    points[index + 2] * (1 - 3 * t) +
                    points[index + 3] * (t)
                );
    }

    inline InterpolationType computeWiggle(size_t index) const
    {
        return floating_t(3) * (points[index + 1] - points[index + 2]) + (points[index + 3] - points[index]);
    }

private: //data
    std::vector<InterpolationType> points;
};




template<class InterpolationType, typename floating_t=float>
class UniformCubicBSpline final : public SplineImpl<UniformCubicBSplineCommon, InterpolationType, floating_t>
{
public:
    UniformCubicBSpline(const std::vector<InterpolationType> &points)
        :SplineImpl<UniformCubicBSplineCommon, InterpolationType,floating_t>(points, points.size() - 3)
    {
        assert(points.size() >= 4);

        this->common = UniformCubicBSplineCommon<InterpolationType, floating_t>(points);
    }
};



template<class InterpolationType, typename floating_t=float>
class LoopingUniformCubicBSpline final : public SplineLoopingImpl<UniformCubicBSplineCommon, InterpolationType, floating_t>
{
public:
    LoopingUniformCubicBSpline(const std::vector<InterpolationType> &points)
        :SplineLoopingImpl<UniformCubicBSplineCommon, InterpolationType,floating_t>(points, points.size())
    {
        size_t degree = 3;

        assert(points.size() >= degree);

        //we need enough space to repeat the last 'degree' elements
        std::vector<InterpolationType> positions(points.size() + degree);

        //it would be easiest to just copy the points vector to the position vector, then copy the first 'degree' elements again
        //this DOES work, but interpolation begins in the wrong place (ie getPosition(0) occurs at the wrong place on the spline)
        //to fix this, we effectively "rotate" the position vector backwards, by copying point[size-1] to the beginning
        //then copying the points vector in after, then copying degree-1 elements from the beginning
        positions[0] = points[points.size() - 1];
        std::copy(points.begin(), points.end(), positions.begin() + 1);
        std::copy_n(points.begin(), degree - 1, positions.end() - (degree - 1));

        this->common = UniformCubicBSplineCommon<InterpolationType, floating_t>(positions);
    }
};
