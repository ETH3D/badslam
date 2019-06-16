#pragma once

#include <cassert>

#include "../spline.h"
#include "../utils/linearalgebra.h"

template<class InterpolationType, typename floating_t>
class NaturalSplineCommon
{
public:
    struct alignas(16) NaturalSplineSegment
    {
        InterpolationType a, c;
    };

    inline NaturalSplineCommon(void) = default;
    inline NaturalSplineCommon(std::vector<NaturalSplineSegment> segments, std::vector<floating_t> knots)
        :segments(std::move(segments)), knots(std::move(knots))
    {}

    inline size_t segmentCount(void) const
    {
        return segments.size() - 1;
    }

    inline size_t segmentForT(floating_t t) const
    {
        size_t segmentIndex = SplineCommon::getIndexForT(knots, t);
        if(segmentIndex >= segmentCount())
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
        size_t segmentIndex = SplineCommon::getIndexForT(knots, globalT);
        if(segmentIndex >= knots.size() - 1)
            segmentIndex--;

        floating_t localT = globalT - knots[segmentIndex];
        floating_t tDiff = knots[segmentIndex + 1] - knots[segmentIndex];

        return computePosition(segmentIndex, tDiff, localT);
    }

    inline typename Spline<InterpolationType,floating_t>::InterpolatedPT getTangent(floating_t globalT) const
    {
        size_t segmentIndex = SplineCommon::getIndexForT(knots, globalT);
        if(segmentIndex >= knots.size() - 1)
            segmentIndex--;

        floating_t localT = globalT - knots[segmentIndex];
        floating_t tDiff = knots[segmentIndex + 1] - knots[segmentIndex];

        return typename Spline<InterpolationType,floating_t>::InterpolatedPT(
                    computePosition(segmentIndex, tDiff, localT),
                    computeTangent(segmentIndex, tDiff, localT)
                    );
    }

    inline typename Spline<InterpolationType,floating_t>::InterpolatedPTC getCurvature(floating_t globalT) const
    {
        size_t segmentIndex = SplineCommon::getIndexForT(knots, globalT);
        if(segmentIndex >= knots.size() - 1)
            segmentIndex--;

        floating_t localT = globalT - knots[segmentIndex];
        floating_t tDiff = knots[segmentIndex + 1] - knots[segmentIndex];

        return typename Spline<InterpolationType,floating_t>::InterpolatedPTC(
                    computePosition(segmentIndex, tDiff, localT),
                    computeTangent(segmentIndex, tDiff, localT),
                    computeCurvature(segmentIndex, tDiff, localT)
                    );
    }

    inline typename Spline<InterpolationType,floating_t>::InterpolatedPTCW getWiggle(floating_t globalT) const
    {
        size_t segmentIndex = SplineCommon::getIndexForT(knots, globalT);
        if(segmentIndex >= knots.size() - 1)
            segmentIndex--;

        floating_t localT = globalT - knots[segmentIndex];
        floating_t tDiff = knots[segmentIndex + 1] - knots[segmentIndex];

        return typename Spline<InterpolationType,floating_t>::InterpolatedPTCW(
                    computePosition(segmentIndex, tDiff, localT),
                    computeTangent(segmentIndex, tDiff, localT),
                    computeCurvature(segmentIndex, tDiff, localT),
                    computeWiggle(segmentIndex, tDiff)
                    );
    }

    inline floating_t segmentLength(size_t segmentIndex, floating_t a, floating_t b) const {

        floating_t tDiff = knots[segmentIndex + 1] - knots[segmentIndex];
        auto segmentFunction = [=](floating_t t) -> floating_t {
            auto tangent = computeTangent(segmentIndex, tDiff, t);
            return tangent.length();
        };

        floating_t localA = a - knots[segmentIndex];
        floating_t localB = b - knots[segmentIndex];

        return SplineLibraryCalculus::gaussLegendreQuadratureIntegral<floating_t>(segmentFunction, localA, localB);
    }

private: //methods
    inline InterpolationType computePosition(size_t index, floating_t tDiff, floating_t t) const
    {
        auto b = computeB(index, tDiff);
        auto d = computeD(index, tDiff);

        return segments[index].a + t * (b + t * (segments[index].c + t * d));
    }

    inline InterpolationType computeTangent(size_t index, floating_t tDiff, floating_t t) const
    {
        auto b = computeB(index, tDiff);
        auto d = computeD(index, tDiff);

        //compute the derivative of the position function
        return b + t * (floating_t(2) * segments[index].c + (3 * t) * d);
    }

    inline InterpolationType computeCurvature(size_t index, floating_t tDiff, floating_t t) const
    {
        auto d = computeD(index, tDiff);

        //compute the 2nd derivative of the position function
        return floating_t(2) * segments[index].c + (6 * t) * d;
    }

    inline InterpolationType computeWiggle(size_t index, floating_t tDiff) const
    {
        auto d = computeD(index, tDiff);

        //compute the 3rd derivative of the position function
        return floating_t(6) * d;
    }


    //B is the tangent at t=0 for a segment, and D is effectively the wiggle for a segment
    //we COULD precompute these and store them alongside a and c in the segment
    //testing shows that it's faster (because of cache, and pipelining, etc) to just recompute them every time
    inline InterpolationType computeB(size_t index, floating_t tDiff) const
    {
        return (segments[index+1].a - segments[index].a) / tDiff - (tDiff / 3) * (segments[index+1].c + floating_t(2)*segments[index].c);
    }
    inline InterpolationType computeD(size_t index, floating_t tDiff) const
    {
        return (segments[index+1].c - segments[index].c) / (3 * tDiff);
    }



    inline floating_t computeSegmentLength(size_t index, floating_t from, floating_t to) const
    {

    }

private: //data
    std::vector<NaturalSplineSegment> segments;
    std::vector<floating_t> knots;
};


template<class InterpolationType, typename floating_t=float>
class NaturalSpline final : public SplineImpl<NaturalSplineCommon, InterpolationType, floating_t>
{
public:
    enum EndConditions { Natural, NotAKnot };

//constructors
public:
    NaturalSpline(const std::vector<InterpolationType> &points,
                  bool includeEndpoints = true,
                  floating_t alpha = 0.0,
                  EndConditions endConditions = Natural)
        :SplineImpl<NaturalSplineCommon, InterpolationType,floating_t>(points, includeEndpoints ? points.size()- 1 : points.size() - 3)
    {
        size_t size = points.size();
        size_t firstPoint;
        size_t numSegments;

        if(includeEndpoints)
        {
            assert(points.size() >= 3);
            numSegments = size - 1;
            firstPoint = 0;
        }
        else
        {
            assert(points.size() >= 4);
            numSegments = size - 3;
            firstPoint = 1;
        }

        //compute the T values for each point
        std::vector<floating_t> paddedKnots = SplineCommon::computeTValuesWithInnerPadding(points, alpha, firstPoint);

        //next we compute curvatures
        std::vector<InterpolationType> curvatures;
        if(endConditions == Natural)
            curvatures = computeCurvaturesNatural(paddedKnots);
        else
            curvatures = computeCurvaturesNotAKnot(paddedKnots);

        //we now have 0 curvature for index 0 and n - 1, and the final (usually nonzero) curvature for every other point
        //use this curvature to determine a,b,c,and d to build each segment
        std::vector<floating_t> knots(numSegments + 1);
        std::vector<typename NaturalSplineCommon<InterpolationType, floating_t>::NaturalSplineSegment> segments(numSegments + 1);
        for(size_t i = firstPoint; i < numSegments + firstPoint + 1; i++) {

            knots[i - firstPoint] = paddedKnots[i];
            segments[i - firstPoint].a = points.at(i);
            segments[i - firstPoint].c = curvatures.at(i);
        }

        this->common = NaturalSplineCommon<InterpolationType, floating_t>(std::move(segments), std::move(knots));
    }
private:
    std::vector<InterpolationType> computeCurvaturesNatural(const std::vector<floating_t> tValues) const;
    std::vector<InterpolationType> computeCurvaturesNotAKnot(const std::vector<floating_t> tValues) const;
};

template<class InterpolationType, typename floating_t=float>
class LoopingNaturalSpline final : public SplineLoopingImpl<NaturalSplineCommon, InterpolationType, floating_t>
{
//constructors
public:
    LoopingNaturalSpline(const std::vector<InterpolationType> &points, floating_t alpha = 0.0)
        :SplineLoopingImpl<NaturalSplineCommon, InterpolationType,floating_t>(points, points.size())
    {
        size_t size = points.size();

        //compute the T values for each point
        std::vector<floating_t> knots = SplineCommon::computeLoopingTValues(points, alpha, 0);

        //now that we know the t values, we need to prepare the tridiagonal matrix calculation
        //note that there several ways to formulate this matrix - i chose the following:
        // http://www-hagen.informatik.uni-kl.de/~alggeom/pdf/ws1213/alggeom_script_ws12_02.pdf

        //the tridiagonal matrix's main diagonal will be neighborDeltaT, and the secondary diagonals will be deltaT
        //the list of values to solve for will be neighborDeltaPoint

        //create an array of the differences in T between one point and the next
        std::vector<floating_t> upperDiagonal(size);
        for(size_t i = 0; i < size; i++)
        {
            floating_t delta = knots[i + 1] - knots[i];
            upperDiagonal[i] = delta;
        }

        //create an array that stores 2 * (deltaT.at(i - 1) + deltaT.at(i))
        //when i = 0, wrap i - 1 back around to the end of the list
        std::vector<floating_t> diagonal(size);
        for(size_t i = 0; i < size; i++)
        {
            floating_t neighborDelta = 2 * (upperDiagonal.at((i - 1 + size)%size) + upperDiagonal.at(i));
            diagonal[i] = neighborDelta;
        }

        //create an array of displacement between each point, divided by delta t
        std::vector<InterpolationType> deltaPoint(size);
        for(size_t i = 0; i < size; i++)
        {
            InterpolationType displacement = points.at((i + 1)%size) - points.at(i);
            deltaPoint[i] = displacement / upperDiagonal.at(i);
        }

        //create an array that stores 3 * (deltaPoint(i - 1) + deltaPoint(i))
        //when i = 0, wrap i - 1 back around to the end of the list
        std::vector<InterpolationType> inputVector(size);
        for(size_t i = 0; i < size; i++)
        {
            InterpolationType neighborDelta = floating_t(3) * (deltaPoint.at(i) - deltaPoint.at((i - 1 + size) % size));
            inputVector[i] = neighborDelta;
        }

        //solve the cyclic tridiagonal system to get the curvature at each point
        std::vector<InterpolationType> curvatures = LinearAlgebra::solveCyclicSymmetricTridiagonal(
                    std::move(diagonal),
                    std::move(upperDiagonal),
                    std::move(inputVector)
                    );

        //we now have the curvature for every point
        //use this curvature to determine a,b,c,and d to build each segment
        std::vector<typename NaturalSplineCommon<InterpolationType, floating_t>::NaturalSplineSegment> segments(size + 1);
        for(size_t i = 0; i < size + 1; i++)
        {
            segments[i].a = points.at(i%size);
            segments[i].c = curvatures.at(i%size);
        }

        this->common = NaturalSplineCommon<InterpolationType, floating_t>(std::move(segments), std::move(knots));
    }
};

template<class InterpolationType, typename floating_t>
std::vector<InterpolationType> NaturalSpline<InterpolationType,floating_t>::computeCurvaturesNatural(const std::vector<floating_t> tValues) const
{

    //now that we know the t values, we need to prepare the tridiagonal matrix calculation
    //note that there several ways to formulate this matrix - for the "natural boundary conditions" i chose the following:
    // http://www-hagen.informatik.uni-kl.de/~alggeom/pdf/ws1213/alggeom_script_ws12_02.pdf

    //the tridiagonal matrix's main diagonal will be neighborDeltaT, and the secondary diagonals will be deltaT
    //the list of values to solve for will be neighborDeltaPoint

    size_t loop_limit = this->getOriginalPoints().size() - 1;

    //create an array of the differences in T between one point and the next
    std::vector<floating_t> upperDiagonal(loop_limit);
    for(size_t i = 0; i < loop_limit; i++)
    {
        floating_t delta = tValues[i + 1] - tValues[i];
        upperDiagonal[i] = delta;
    }

    //create an array that stores 2 * (deltaT.at(i - 1) + deltaT.at(i))
    std::vector<floating_t> diagonal(loop_limit - 1);
    for(size_t i = 1; i < loop_limit; i++)
    {
        floating_t neighborDelta = floating_t(2) * (upperDiagonal[i - 1] + upperDiagonal[i]);
        diagonal[i - 1] = neighborDelta;
    }

    //create an array of displacement between each point, divided by delta t
    std::vector<InterpolationType> deltaPoint(loop_limit);
    for(size_t i = 0; i < loop_limit; i++)
    {
        InterpolationType displacement = this->getOriginalPoints()[i + 1] - this->getOriginalPoints()[i];
        deltaPoint[i] = displacement / upperDiagonal[i];
    }

    //create an array that stores 3 * (deltaPoint(i - 1) + deltaPoint(i))
    std::vector<InterpolationType> inputVector(loop_limit - 1);
    for(size_t i = 1; i < loop_limit; i++)
    {
        InterpolationType neighborDelta = floating_t(3) * (deltaPoint[i] - deltaPoint[i - 1]);
        inputVector[i - 1] = neighborDelta;
    }

    //the first element in upperDiagonal is garbage, so remove it
    upperDiagonal.erase(upperDiagonal.begin());

    //solve the tridiagonal system to get the curvature at each point
    std::vector<InterpolationType> curvatures = LinearAlgebra::solveSymmetricTridiagonal(
                std::move(diagonal),
                std::move(upperDiagonal),
                std::move(inputVector)
                );

    //we didn't compute the first or last curvature, which will be 0
    curvatures.insert(curvatures.begin(), InterpolationType());
    curvatures.push_back(InterpolationType());

    return curvatures;
}


template<class InterpolationType, typename floating_t>
std::vector<InterpolationType> NaturalSpline<InterpolationType,floating_t>::computeCurvaturesNotAKnot(const std::vector<floating_t> tValues) const
{
    //now that we know the t values, we need to prepare the tridiagonal matrix calculation
    //note that there several ways to formulate this matrix; for "not a knot" i chose the following:
    // http://sepwww.stanford.edu/data/media/public/sep//sergey/128A/answers6.pdf

    //the tridiagonal matrix's main diagonal will be neighborDeltaT, and the secondary diagonals will be deltaT
    //the list of values to solve for will be neighborDeltaPoint

    size_t size = this->getOriginalPoints().size() - 1;

    //create an array of the differences in T between one point and the next
    std::vector<floating_t> deltaT(size);
    for(size_t i = 0; i < size; i++)
    {
        deltaT[i] = tValues[i + 1] - tValues[i];
    }

    //the main diagonal of the tridiagonal will be 2 * (deltaT[i] + deltaT[i + 1])
    float mainDiagonalSize = size - 1;
    std::vector<floating_t> mainDiagonal(mainDiagonalSize);
    for(size_t i = 0; i < mainDiagonalSize; i++)
    {
        mainDiagonal[i] = 2 * (deltaT[i] + deltaT[i + 1]);
    }

    //the upper diagonal will just be deltaT[i + 1]
    float secondaryDiagonalSize = size - 2;
    std::vector<floating_t> upperDiagonal(secondaryDiagonalSize);
    for(size_t i = 0; i < secondaryDiagonalSize; i++)
    {
        upperDiagonal[i] = deltaT[i + 1];
    }

    //the lower diagonal is just a copy of the upper diagonal
    std::vector<floating_t> lowerDiagonal = upperDiagonal;

    //create an array of displacement between each point, divided by delta t
    std::vector<InterpolationType> deltaPoint(size);
    for(size_t i = 0; i < size; i++)
    {
        InterpolationType displacement = this->getOriginalPoints()[i + 1] - this->getOriginalPoints()[i];
        deltaPoint[i] = displacement / deltaT[i];
    }

    //create an array that stores 3 * (deltaPoint(i - 1) + deltaPoint(i))
    std::vector<InterpolationType> inputVector(mainDiagonalSize);
    for(size_t i = 0; i < mainDiagonalSize; i++)
    {
        inputVector[i] = floating_t(3) * (deltaPoint[i + 1] - deltaPoint[i]);
    }

    //the first and last of the values in maindiagonalare different than normal
    mainDiagonal[0] = 3*deltaT[0] + 2*deltaT[1] + deltaT[0]*deltaT[0]/deltaT[1];
    mainDiagonal[mainDiagonalSize - 1] = 3*deltaT[size - 1] + 2*deltaT[size - 2] + deltaT[size - 1]*deltaT[size - 1]/deltaT[size - 2];

    //the first value in the upper diagonal is different than normal
    upperDiagonal[0] = deltaT[1] - deltaT[0]*deltaT[0]/deltaT[1];

    //the last value in the upper diagonal is different than normal
    lowerDiagonal[secondaryDiagonalSize - 1] = deltaT[size - 2] - deltaT[size - 1]*deltaT[size - 1]/deltaT[size - 2];

    //solve the tridiagonal system to get the curvature at each point
    std::vector<InterpolationType> curvatures = LinearAlgebra::solveTridiagonal(
                std::move(mainDiagonal),
                std::move(upperDiagonal),
                std::move(lowerDiagonal),
                std::move(inputVector)
                );

    //we didn't compute the first or last curvature, which will be calculated based on the others
    curvatures.insert(curvatures.begin(), curvatures[0] * (1 + deltaT[0]/deltaT[1]) - curvatures[1] * (deltaT[0]/deltaT[1]));
    curvatures.push_back(curvatures.back() * (1 + deltaT[size - 1]/deltaT[size - 2])
            - curvatures[curvatures.size() - 2] * (deltaT[size - 1]/deltaT[size - 2]));

    return curvatures;
}
