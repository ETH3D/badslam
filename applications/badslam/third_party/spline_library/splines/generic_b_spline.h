#pragma once

#include <cassert>

#include "../spline.h"

template<class InterpolationType, typename floating_t>
class GenericBSplineCommon
{
public:
    inline GenericBSplineCommon(void) = default;
    inline GenericBSplineCommon(std::vector<InterpolationType> positions, std::vector<floating_t> knots, size_t splineDegree)
        :positions(std::move(positions)), knots(std::move(knots)), splineDegree(splineDegree)
    {}

    inline size_t segmentCount(void) const
    {
        return positions.size() - splineDegree;
    }

    inline size_t segmentForT(floating_t t) const
    {
        if(t < 0) {
            return 0;
        }

        size_t segmentIndex = SplineCommon::getIndexForT(knots, t) - (splineDegree - 1);
        if(segmentIndex > segmentCount() - 1)
        {
            return segmentCount() - 1;
        }
        else
        {
            return segmentIndex;
        }
    }

    inline floating_t segmentT(size_t segmentIndex) const
    {
        return knots[segmentIndex + splineDegree - 1];
    }

    inline InterpolationType getPosition(floating_t globalT) const
    {
        size_t segmentIndex = segmentForT(globalT);
        size_t innerIndex = segmentIndex + (splineDegree - 1);

        return computeDeboor(innerIndex + 1, splineDegree, globalT);
    }

    inline typename Spline<InterpolationType,floating_t>::InterpolatedPT getTangent(floating_t globalT) const
    {
        size_t segmentIndex = segmentForT(globalT);
        size_t innerIndex = segmentIndex + (splineDegree - 1);

        return typename Spline<InterpolationType,floating_t>::InterpolatedPT(
                    computeDeboor(innerIndex + 1, splineDegree, globalT),
                    computeDeboorDerivative(innerIndex + 1, splineDegree, globalT, 1)
                    );
    }

    inline typename Spline<InterpolationType,floating_t>::InterpolatedPTC getCurvature(floating_t globalT) const
    {
        size_t segmentIndex = segmentForT(globalT);
        size_t innerIndex = segmentIndex + (splineDegree - 1);

        return typename Spline<InterpolationType,floating_t>::InterpolatedPTC(
                    computeDeboor(innerIndex + 1, splineDegree, globalT),
                    computeDeboorDerivative(innerIndex + 1, splineDegree, globalT, 1),
                    computeDeboorDerivative(innerIndex + 1, splineDegree, globalT, 2)
                    );
    }

    inline typename Spline<InterpolationType,floating_t>::InterpolatedPTCW getWiggle(floating_t globalT) const
    {
        size_t segmentIndex = segmentForT(globalT);
        size_t innerIndex = segmentIndex + (splineDegree - 1);

        return typename Spline<InterpolationType,floating_t>::InterpolatedPTCW(
                    computeDeboor(innerIndex + 1, splineDegree, globalT),
                    computeDeboorDerivative(innerIndex + 1, splineDegree, globalT, 1),
                    computeDeboorDerivative(innerIndex + 1, splineDegree, globalT, 2),
                    computeDeboorDerivative(innerIndex + 1, splineDegree, globalT, 3)
                    );
    }

    inline floating_t segmentLength(size_t segmentIndex, floating_t a, floating_t b) const {

        auto innerIndex = segmentIndex + splineDegree - 1;

        floating_t tDistance = knots[innerIndex + 1] - knots[innerIndex];

        //it's perfectly legal for Bspline segments to have a T distance of 0, in which case the arc length is 0
        if(tDistance > 0)
        {
            auto segmentFunction = [this, innerIndex](floating_t t) -> floating_t {
                auto tangent = computeDeboorDerivative(innerIndex + 1, splineDegree, t, 1);
                return tangent.length();
            };

            return SplineLibraryCalculus::gaussLegendreQuadratureIntegral<floating_t>(segmentFunction, a, b);
        }
        else
        {
            return 0;
        }
    }

private: //methods
    InterpolationType computeDeboor(size_t knotIndex, size_t degree, float globalT) const;
    InterpolationType computeDeboorDerivative(size_t knotIndex, size_t degree, float globalT, int derivativeLevel) const;

private: //data
    std::vector<InterpolationType> positions;
    std::vector<floating_t> knots;
    size_t splineDegree;
};

template<class InterpolationType, typename floating_t>
InterpolationType GenericBSplineCommon<InterpolationType,floating_t>::computeDeboor(size_t knotIndex, size_t degree, float globalT) const
{
    if(degree == 0)
    {
        return positions[knotIndex];
    }
    else
    {
        floating_t alpha = (globalT - knots[knotIndex - 1]) / (knots[knotIndex + splineDegree - degree] - knots[knotIndex - 1]);

        InterpolationType leftRecursive = computeDeboor(knotIndex - 1, degree - 1, globalT);
        InterpolationType rightRecursive = computeDeboor(knotIndex, degree - 1, globalT);

        InterpolationType blended = leftRecursive * (1 - alpha) + rightRecursive * alpha;

        return blended;
    }
}

template<class InterpolationType, typename floating_t>
InterpolationType GenericBSplineCommon<InterpolationType,floating_t>::computeDeboorDerivative(size_t knotIndex, size_t degree, float globalT, int derivativeLevel) const
{
    if(degree == 0)
    {
        //if we hit degree 0 before derivative level 0, then this spline's
        //degree isn't high enough to support whatever derivative level was requested
        return InterpolationType();
    }
    else
    {
        floating_t multiplier = degree / (knots[knotIndex + splineDegree - degree] - knots[knotIndex - 1]);

        if(derivativeLevel <= 1)
        {
            //once we reach this point, the derivative calculation is "complete"
            //in that from here, we go back to the normal deboor calculation deeper in the recursive tree
            return multiplier *
                    (computeDeboor(knotIndex, degree - 1, globalT)
                   - computeDeboor(knotIndex - 1, degree - 1, globalT)
                     );
        }
        else
        {
            //recursively call the derivative function to compute a higher derivative
            return multiplier *
                    (computeDeboorDerivative(knotIndex, degree - 1, globalT, derivativeLevel - 1)
                   - computeDeboorDerivative(knotIndex - 1, degree - 1, globalT, derivativeLevel - 1)
                     );
        }
    }
}

template<class InterpolationType, typename floating_t=float>
class GenericBSpline final : public SplineImpl<GenericBSplineCommon, InterpolationType, floating_t>
{
//constructors
public:
    GenericBSpline(const std::vector<InterpolationType> &points, size_t degree)
        :SplineImpl<GenericBSplineCommon, InterpolationType,floating_t>(points, points.size() - degree)
    {
        assert(points.size() > degree);

        std::vector<floating_t> knots(points.size() + degree - 1);
        for(size_t i = 0; i < knots.size(); i++)
        {
            knots[i] = floating_t(i) - floating_t(degree - 1);
        }

        this->common = GenericBSplineCommon<InterpolationType, floating_t>(points, std::move(knots), degree);
    }
};

template<class InterpolationType, typename floating_t=float>
class LoopingGenericBSpline final : public SplineLoopingImpl<GenericBSplineCommon, InterpolationType, floating_t>
{
//constructors
public:
    LoopingGenericBSpline(const std::vector<InterpolationType> &points, size_t degree)
        :SplineLoopingImpl<GenericBSplineCommon, InterpolationType,floating_t>(points, points.size())
    {
        assert(points.size() > degree);

        std::vector<floating_t> knots(points.size() + degree * 2 - 1);
        for(size_t i = 0; i < knots.size(); i++)
        {
            knots[i] = floating_t(i) - floating_t(degree - 1);
        }

        //it would be easiest to just copy the points vector to the position vector, then copy the first 'degree' elements again
        //this DOES work, but interpolation begins in the wrong place (ie getPosition(0) occurs at the wrong place on the spline)
        //to fix this, we effectively "rotate" the position vector backwards, by copying point[size-1] to the beginning
        //then copying the points vector in after, then copying degree-1 elements from the beginning
        std::vector<InterpolationType> positions(points.size() + degree);

        size_t padding = degree - 1;
        positions[0] = points[points.size() - 1];
        std::copy(points.begin(), points.end(), positions.begin() + 1);
        std::copy_n(points.begin(), padding, positions.end() - padding);

        this->common = GenericBSplineCommon<InterpolationType, floating_t>(std::move(positions), std::move(knots), degree);
    }
};

