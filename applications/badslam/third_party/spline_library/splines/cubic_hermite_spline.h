#pragma once

#include <cassert>

#include "../spline.h"

template<class InterpolationType, typename floating_t>
class CubicHermiteSplineCommon
{
public:
    struct alignas(8) CubicHermiteSplinePoint
    {
        InterpolationType position, tangent;
    };

    inline CubicHermiteSplineCommon(void) = default;
    inline CubicHermiteSplineCommon(std::vector<CubicHermiteSplinePoint> points, std::vector<floating_t> knots)
        :points(std::move(points)), knots(std::move(knots))
    {}

    inline size_t segmentCount(void) const
    {
        return points.size() - 1;
    }

    inline size_t segmentForT(floating_t t) const
    {
        size_t segmentIndex = SplineCommon::getIndexForT(knots, t);
        if(segmentIndex > segmentCount() - 1)
            return segmentCount() - 1;
        else
            return segmentIndex;
    }

    inline floating_t segmentT(size_t segmentIndex) const
    {
        return knots[segmentIndex];
    }

    inline InterpolationType getPosition(floating_t globalT) const
    {
        size_t knotIndex = segmentForT(globalT);

        floating_t tDiff = (knots[knotIndex + 1] - knots[knotIndex]);
        floating_t localT = (globalT - knots[knotIndex]) / tDiff;

        return computePosition(knotIndex, tDiff, localT);
    }

    inline typename Spline<InterpolationType,floating_t>::InterpolatedPT getTangent(floating_t globalT) const
    {
        size_t knotIndex = segmentForT(globalT);

        floating_t tDiff = (knots[knotIndex + 1] - knots[knotIndex]);
        floating_t localT = (globalT - knots[knotIndex]) / tDiff;

        return typename Spline<InterpolationType,floating_t>::InterpolatedPT(
                    computePosition(knotIndex, tDiff, localT),
                    computeTangent(knotIndex, tDiff, localT)
                    );
    }

    inline typename Spline<InterpolationType,floating_t>::InterpolatedPTC getCurvature(floating_t globalT) const
    {
        size_t knotIndex = segmentForT(globalT);

        floating_t tDiff = (knots[knotIndex + 1] - knots[knotIndex]);
        floating_t localT = (globalT - knots[knotIndex]) / tDiff;

        return typename Spline<InterpolationType,floating_t>::InterpolatedPTC(
                    computePosition(knotIndex, tDiff, localT),
                    computeTangent(knotIndex, tDiff, localT),
                    computeCurvature(knotIndex, tDiff, localT)
                    );
    }

    inline typename Spline<InterpolationType,floating_t>::InterpolatedPTCW getWiggle(floating_t globalT) const
    {
        size_t knotIndex = segmentForT(globalT);

        floating_t tDiff = knots[knotIndex + 1] - knots[knotIndex];
        floating_t localT = (globalT - knots[knotIndex]) / tDiff;

        return typename Spline<InterpolationType,floating_t>::InterpolatedPTCW(
                    computePosition(knotIndex, tDiff, localT),
                    computeTangent(knotIndex, tDiff, localT),
                    computeCurvature(knotIndex, tDiff, localT),
                    computeWiggle(knotIndex, tDiff)
                    );
    }

    inline floating_t segmentLength(size_t index, floating_t a, floating_t b) const
    {
        floating_t tDiff = knots[index + 1] - knots[index];
        auto segmentFunction = [this, index, tDiff](floating_t t) -> floating_t {
            auto tangent = computeTangent(index, tDiff, t);
            return tangent.length();
        };

        floating_t localA = (a - knots[index]) / tDiff;
        floating_t localB = (b - knots[index]) / tDiff;

        return tDiff * SplineLibraryCalculus::gaussLegendreQuadratureIntegral<floating_t>(segmentFunction, localA, localB);
    }


private: //methods
    inline InterpolationType computePosition(size_t index, floating_t tDiff, floating_t t) const
    {
        auto oneMinusT = 1 - t;

        auto basis00 = (1 + 2*t) * oneMinusT * oneMinusT;
        auto basis10 = t * oneMinusT * oneMinusT;

        auto basis11 = t * t * -oneMinusT;
        auto basis01 = t * t * (3 - 2*t);

        return
                basis00 * points[index].position +
                basis10 * tDiff * points[index].tangent +

                basis11 * tDiff * points[index + 1].tangent +
                basis01 * points[index + 1].position;
    }

    inline InterpolationType computeTangent(size_t index, floating_t tDiff, floating_t t) const
    {
        auto oneMinusT = 1 - t;

        auto d_basis00 = 6 * t * (t - 1);
        auto d_basis10 = (1 - 3*t) * oneMinusT;

        auto d_basis11 = t * (3 * t - 2);
        auto d_basis01 = -d_basis00;

        //tests and such have shown that we have to scale this by the inverse of the t distance, and i'm not sure why
        //intuitively it would just be the derivative of the position function and nothing else
        //if you know why please let me know
        return (
                d_basis00 * points[index].position +
                d_basis10 * tDiff * points[index].tangent +

                d_basis11 * tDiff * points[index + 1].tangent +
                d_basis01 * points[index + 1].position
                ) / tDiff;
    }

    inline InterpolationType computeCurvature(size_t index, floating_t tDiff, floating_t t) const
    {
        auto d2_basis00 = 6 * (2 * t - 1);
        auto d2_basis10 = 2 * (3 * t - 2);

        auto d2_basis11 = 2 * (3 * t - 1);
        auto d2_basis01 = -d2_basis00;

        //tests and such have shown that we have to scale this by the inverse of the t distance, and i'm not sure why
        //intuitively it would just be the 2nd derivative of the position function and nothing else
        //if you know why please let me know
        return (
                d2_basis00 * points[index].position +
                d2_basis10 * tDiff * points[index].tangent +

                d2_basis11 * tDiff * points[index + 1].tangent +
                d2_basis01 * points[index + 1].position
                ) / (tDiff * tDiff);
    }

    inline InterpolationType computeWiggle(size_t index, floating_t tDiff) const
    {
        //tests and such have shown that we have to scale this by the inverse of the t distance, and i'm not sure why
        //intuitively it would just be the 2nd derivative of the position function and nothing else
        //if you know why please let me know
        return (
                    floating_t(12) * (points[index].position - points[index + 1].position) + floating_t(6) * tDiff * (points[index].tangent + points[index + 1].tangent)
                ) / (tDiff * tDiff * tDiff);
    }

private: //data
    std::vector<CubicHermiteSplinePoint> points;
    std::vector<floating_t> knots;
};



template<class InterpolationType, typename floating_t=float>
class CubicHermiteSpline final : public SplineImpl<CubicHermiteSplineCommon, InterpolationType, floating_t>
{
//constructors
public:
    CubicHermiteSpline(const std::vector<InterpolationType> &points, const std::vector<InterpolationType> &tangents, floating_t alpha = 0.0)
        :SplineImpl<CubicHermiteSplineCommon, InterpolationType,floating_t>(points, points.size() - 1)
    {
        assert(points.size() >= 2);
        assert(points.size() == tangents.size());

        size_t firstTangent = 0;
        size_t numSegments = points.size() - 1;

        //compute the T values for each point
        std::vector<floating_t> knots = SplineCommon::computeTValuesWithInnerPadding(points, alpha, firstTangent);

        //pre-arrange the data needed for interpolation
        std::vector<typename CubicHermiteSplineCommon<InterpolationType, floating_t>::CubicHermiteSplinePoint> positionData(numSegments + 1);
        for(size_t i = 0; i < positionData.size(); i++)
        {
            positionData[i].position = points[i];
            positionData[i].tangent = tangents[i];
        }

        this->common = CubicHermiteSplineCommon<InterpolationType, floating_t>(std::move(positionData), std::move(knots));
    }

    CubicHermiteSpline(const std::vector<InterpolationType> &points, floating_t alpha = 0.0)
        :SplineImpl<CubicHermiteSplineCommon, InterpolationType,floating_t>(points, points.size() - 3)
    {
        assert(points.size() >= 4);

        size_t firstTangent = 1;
        size_t numSegments = points.size() - 3;

        //compute the T values for each point
        std::vector<floating_t> paddedKnots = SplineCommon::computeTValuesWithInnerPadding(points, alpha, firstTangent);

        //compute the tangents
        std::vector<InterpolationType> tangents(points.size());
        for(size_t i = firstTangent; i < firstTangent + numSegments + 1; i++)
        {
            floating_t tPrev = paddedKnots[i - 1];
            floating_t tCurrent = paddedKnots[i];
            floating_t tNext = paddedKnots[i + 1];

            InterpolationType pPrev = points.at(i - 1);
            InterpolationType pCurrent = points.at(i);
            InterpolationType pNext = points.at(i + 1);

            //the tangent is the standard catmull-rom spline tangent calculation
            tangents[i] =
                                pPrev * (tCurrent - tNext) / ((tNext - tPrev) * (tCurrent - tPrev))
                                + pNext * (tCurrent - tPrev) / ((tNext - tPrev) * (tNext - tCurrent))

                             //plus a little something extra - this is derived from the pyramid contruction
                             //when the t values are evenly spaced (ie when alpha is 0), this whole line collapses to 0,
                             //yielding the standard catmull-rom formula
                                - pCurrent * ((tCurrent - tPrev) - (tNext - tCurrent)) / ((tNext - tCurrent) * (tCurrent - tPrev));
        }

        //pre-arrange the data needed for interpolation
        std::vector<floating_t> knots(numSegments + 1);
        std::vector<typename CubicHermiteSplineCommon<InterpolationType, floating_t>::CubicHermiteSplinePoint> positionData(numSegments + 1);
        for(size_t i = 0; i < positionData.size(); i++)
        {
            knots[i] = paddedKnots[i + firstTangent];
            positionData[i].position = points[i + firstTangent];
            positionData[i].tangent = tangents[i + firstTangent];
        }

        this->common = CubicHermiteSplineCommon<InterpolationType, floating_t>(std::move(positionData), std::move(knots));
    }
};




template<class InterpolationType, typename floating_t=float>
class LoopingCubicHermiteSpline final : public SplineLoopingImpl<CubicHermiteSplineCommon, InterpolationType, floating_t>
{
//constructors
public:
    LoopingCubicHermiteSpline(const std::vector<InterpolationType> &points, const std::vector<InterpolationType> &tangents, floating_t alpha = 0.0)
        :SplineLoopingImpl<CubicHermiteSplineCommon, InterpolationType,floating_t>(points, points.size())
    {
        assert(points.size() >= 2);
        assert(points.size() == tangents.size());

        //compute the T values for each point
        std::vector<floating_t> knots = SplineCommon::computeLoopingTValues(points, alpha, 0);

        //pre-arrange the data needed for interpolation
        std::vector<typename CubicHermiteSplineCommon<InterpolationType, floating_t>::CubicHermiteSplinePoint> positionData(points.size() + 1);
        for(size_t i = 0; i < points.size(); i++)
        {
            positionData[i].position = points[i];
            positionData[i].tangent = tangents[i];
        }
        positionData[points.size()] = positionData[0];

        this->common = CubicHermiteSplineCommon<InterpolationType, floating_t>(std::move(positionData), std::move(knots));
    }

    LoopingCubicHermiteSpline(const std::vector<InterpolationType> &points, floating_t alpha = 0.0)
        :SplineLoopingImpl<CubicHermiteSplineCommon, InterpolationType,floating_t>(points, points.size())
    {
        assert(points.size() >= 4);

        size_t size = points.size();

        //compute the T values for each point
        size_t padding = 1;
        std::vector<floating_t> paddedKnots = SplineCommon::computeLoopingTValues(points, alpha, padding);

        //compute the tangents
        std::vector<InterpolationType> tangents(size);
        for(size_t i = 0; i < size; i++)
        {
            floating_t tPrev = paddedKnots[i - 1 + padding];
            floating_t tCurrent = paddedKnots[i + padding];
            floating_t tNext = paddedKnots[i + 1 + padding];

            InterpolationType pPrev = points[(i - 1 + size)%size];
            InterpolationType pCurrent = points[i];
            InterpolationType pNext = points[(i + 1)%size];

            //the tangent is the standard catmull-rom spline tangent calculation
            tangents[i] =
                                pPrev * (tCurrent - tNext) / ((tNext - tPrev) * (tCurrent - tPrev))
                                + pNext * (tCurrent - tPrev) / ((tNext - tPrev) * (tNext - tCurrent))

                             //plus a little something extra - this is derived from the pyramid contruction
                             //when the t values are evenly spaced (ie when alpha is 0), this whole line collapses to 0,
                             //yielding the standard catmull-rom formula
                                - pCurrent * ((tCurrent - tPrev) - (tNext - tCurrent)) / ((tNext - tCurrent) * (tCurrent - tPrev));
        }

        //pre-arrange the data needed for interpolation
        std::vector<floating_t> knots(size + 1);
        std::vector<typename CubicHermiteSplineCommon<InterpolationType, floating_t>::CubicHermiteSplinePoint> positionData(size + 1);
        for(size_t i = 0; i < size; i++)
        {
            knots[i] = paddedKnots[i + padding];
            positionData[i].position = points[i];
            positionData[i].tangent = tangents[i];
        }
        positionData[size] = positionData[0];
        knots[size] = paddedKnots[size + padding];

        this->common = CubicHermiteSplineCommon<InterpolationType, floating_t>(std::move(positionData), std::move(knots));
    }
};
