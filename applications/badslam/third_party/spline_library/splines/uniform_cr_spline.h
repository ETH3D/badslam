#pragma once

#include <cassert>

#include "../spline.h"

template<class InterpolationType, typename floating_t>
class UniformCRSplineCommon
{
public:

    inline UniformCRSplineCommon(void) = default;
    inline UniformCRSplineCommon(std::vector<InterpolationType> points)
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

        return computePosition(segmentIndex + 1, localT);
    }

    inline typename Spline<InterpolationType,floating_t>::InterpolatedPT getTangent(floating_t globalT) const
    {
        size_t segmentIndex = segmentForT(globalT);
        floating_t localT = globalT - segmentIndex;

        return typename Spline<InterpolationType,floating_t>::InterpolatedPT(
                    computePosition(segmentIndex + 1, localT),
                    computeTangent(segmentIndex + 1, localT)
                    );
    }

    inline typename Spline<InterpolationType,floating_t>::InterpolatedPTC getCurvature(floating_t globalT) const
    {
        size_t segmentIndex = segmentForT(globalT);
        floating_t localT = globalT - segmentIndex;

        return typename Spline<InterpolationType,floating_t>::InterpolatedPTC(
                    computePosition(segmentIndex + 1, localT),
                    computeTangent(segmentIndex + 1, localT),
                    computeCurvature(segmentIndex + 1, localT)
                    );
    }

    inline typename Spline<InterpolationType,floating_t>::InterpolatedPTCW getWiggle(floating_t globalT) const
    {
        size_t segmentIndex = segmentForT(globalT);
        floating_t localT = globalT - segmentIndex;

        return typename Spline<InterpolationType,floating_t>::InterpolatedPTCW(
                    computePosition(segmentIndex + 1, localT),
                    computeTangent(segmentIndex + 1, localT),
                    computeCurvature(segmentIndex + 1, localT),
                    computeWiggle(segmentIndex + 1)
                    );
    }

    inline floating_t segmentLength(size_t index, floating_t a, floating_t b) const
    {
        auto segmentFunction = [this, index](floating_t t) -> floating_t {
            auto tangent = computeTangent(index + 1, t);
            return tangent.length();
        };

        floating_t localA = a - index;
        floating_t localB = b - index;

        return SplineLibraryCalculus::gaussLegendreQuadratureIntegral<floating_t>(segmentFunction, localA, localB);
    }


private: //methods
    inline InterpolationType computePosition(size_t index, floating_t t) const
    {
        auto beforeTangent = computeTangentAtIndex(index);
        auto afterTangent = computeTangentAtIndex(index + 1);

        auto oneMinusT = 1 - t;

        auto basis00 = (1 + 2*t) * oneMinusT * oneMinusT;
        auto basis10 = t * oneMinusT * oneMinusT;

        auto basis11 = t * t * -oneMinusT;
        auto basis01 = t * t * (3 - 2*t);

        return
                basis00 * points[index] +
                basis10 * beforeTangent +

                basis11 * afterTangent +
                basis01 * points[index + 1];
    }

    inline InterpolationType computeTangent(size_t index, floating_t t) const
    {
        auto beforeTangent = computeTangentAtIndex(index);
        auto afterTangent = computeTangentAtIndex(index + 1);

        auto oneMinusT = 1 - t;

        auto d_basis00 = 6 * t * (t - 1);
        auto d_basis10 = (1 - 3*t) * oneMinusT;

        auto d_basis11 = t * (3 * t - 2);
        auto d_basis01 = -d_basis00;

        //tests and such have shown that we have to scale this by the inverse of the t distance, and i'm not sure why
        //intuitively it would just be the derivative of the position function and nothing else
        //if you know why please let me know
        return
                d_basis00 * points[index] +
                d_basis10 * beforeTangent +

                d_basis11 * afterTangent +
                d_basis01 * points[index + 1];
    }

    inline InterpolationType computeCurvature(size_t index, floating_t t) const
    {
        auto beforeTangent = computeTangentAtIndex(index);
        auto afterTangent = computeTangentAtIndex(index + 1);

        auto d2_basis00 = 6 * (2 * t - 1);
        auto d2_basis10 = 2 * (3 * t - 2);

        auto d2_basis11 = 2 * (3 * t - 1);
        auto d2_basis01 = -d2_basis00;

        //tests and such have shown that we have to scale this by the inverse of the t distance, and i'm not sure why
        //intuitively it would just be the 2nd derivative of the position function and nothing else
        //if you know why please let me know
        return
                d2_basis00 * points[index] +
                d2_basis10 * beforeTangent +

                d2_basis11 * afterTangent +
                d2_basis01 * points[index + 1];
    }

    inline InterpolationType computeWiggle(size_t index) const
    {
        auto beforeTangent = computeTangentAtIndex(index);
        auto afterTangent = computeTangentAtIndex(index + 1);

        //tests and such have shown that we have to scale this by the inverse of the t distance, and i'm not sure why
        //intuitively it would just be the 2nd derivative of the position function and nothing else
        //if you know why please let me know
        return floating_t(12) * (points[index] - points[index + 1]) + floating_t(6) * (beforeTangent + afterTangent);
    }

    inline InterpolationType computeTangentAtIndex(size_t i) const
    {
        return (points[i + 1] - points[i - 1]) / floating_t(2);
    }

private: //data
    std::vector<InterpolationType> points;
};




template<class InterpolationType, typename floating_t=float>
class UniformCRSpline final : public SplineImpl<UniformCRSplineCommon, InterpolationType, floating_t>
{
//constructors
public:
    UniformCRSpline(const std::vector<InterpolationType> &points)
        :SplineImpl<UniformCRSplineCommon, InterpolationType, floating_t>(points, points.size() - 3)
    {
        assert(points.size() >= 4);

        this->common = UniformCRSplineCommon<InterpolationType, floating_t>(points);
    }
};


template<class InterpolationType, typename floating_t=float>
class LoopingUniformCRSpline final : public SplineLoopingImpl<UniformCRSplineCommon, InterpolationType, floating_t>
{
//constructors
public:
    LoopingUniformCRSpline(const std::vector<InterpolationType> &points)
        :SplineLoopingImpl<UniformCRSplineCommon, InterpolationType,floating_t>(points, points.size())
    {
        assert(points.size() >= 4);

        //we need enough space to repeat the last 'degree' elements
        std::vector<InterpolationType> positions(points.size() + 3);

        //it would be easiest to just copy the points vector to the position vector, then copy the first 'degree' elements again
        //this DOES work, but interpolation begins in the wrong place (ie getPosition(0) occurs at the wrong place on the spline)
        //to fix this, we effectively "rotate" the position vector backwards, by copying point[size-1] to the beginning
        //then copying the points vector in after, then copying degree-1 elements from the beginning
        positions[0] = points.back();
        std::copy(points.begin(), points.end(), positions.begin() + 1);
        std::copy_n(points.begin(), 2, positions.end() - 2);

        this->common = UniformCRSplineCommon<InterpolationType, floating_t>(std::move(positions));
    }
};
