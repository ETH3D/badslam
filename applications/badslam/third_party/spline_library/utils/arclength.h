#pragma once

#include <boost/math/tools/roots.hpp>

#include "spline_common.h"

namespace __ArcLengthSolvePrivate
{
    //solve the arc length for a single spline segment
    template<template <class, typename> class Spline, class InterpolationType, typename floating_t>
    floating_t solveSegment(const Spline<InterpolationType, floating_t>& spline, size_t segmentIndex, floating_t desiredLength, floating_t maxLength, floating_t segmentA)
    {
        //we can use the lengths we've calculated to formulate a pretty solid guess
        //if desired length is x% of the bLength, then our guess will be x% of the way from aPercent to 1
        floating_t desiredPercent = desiredLength / maxLength;
        floating_t bEnd = spline.segmentT(segmentIndex + 1);
        floating_t bGuess = segmentA + desiredPercent * (bEnd - segmentA);

        auto solveFunction = [&](floating_t b) {
            floating_t value = spline.segmentArcLength(segmentIndex, segmentA, b) - desiredLength;

            //the derivative will be the length of the tangent
            auto interpolationResult = spline.getCurvature(b);
            floating_t tangentLength = interpolationResult.tangent.length();

            //the second derivative will be the curvature projected onto the tangent
            interpolationResult.tangent /= tangentLength;
            floating_t secondDerivative = InterpolationType::dotProduct(interpolationResult.tangent, interpolationResult.curvature);

            return std::make_tuple(value, tangentLength, secondDerivative);
        };

        return boost::math::tools::halley_iterate(solveFunction, bGuess, segmentA, bEnd, int(std::numeric_limits<floating_t>::digits * 0.5));
    }
}

namespace ArcLength
{
    //compute b such that arcLength(a,b) == desiredLength
    template<template <class, typename> class SplineT, class InterpolationType, typename floating_t>
    floating_t solveLength(const SplineT<InterpolationType, floating_t>& spline, floating_t a, floating_t desiredLength)
    {
        size_t index = spline.segmentForT(a);

        floating_t segmentLength;
        floating_t segmentBegin = a;

        //scan through the spline's segments until we find the segment that contains b
        do
        {
            segmentLength = spline.segmentArcLength(index, segmentBegin, spline.segmentT(index + 1));

            if(segmentLength < desiredLength)
            {
                index++;
                desiredLength -= segmentLength;
                segmentBegin = spline.segmentT(index);
            }
            else
            {
                break;
            }
        }
        while(index < spline.segmentCount());

        //if bIndex is equal to the segment count, we've hit the end of the spline, so return maxT
        if(index == spline.segmentCount())
        {
            return spline.getMaxT();
        }

        return __ArcLengthSolvePrivate::solveSegment(spline, index, desiredLength, segmentLength, segmentBegin);
    }

    //compute b such that cyclicArcLength(a,b) == desiredLength, respecting the cyclic semantics of a looping spline
    //IE, a can be out of range, if desiredLength is totalLength*2 + 1, the result will be equal to solveCyclic(a,1) + maxT*2
    template<template <class, typename> class LoopingSplineT, class InterpolationType, typename floating_t>
    floating_t solveLengthCyclic(const LoopingSplineT<InterpolationType, floating_t>& spline, floating_t a, floating_t desiredLength)
    {
        size_t index = spline.segmentForT(a);

        floating_t wrappedA = spline.wrapT(a);
        floating_t segmentBegin = wrappedA;
        floating_t segmentLength = spline.segmentArcLength(index, segmentBegin, spline.segmentT(index + 1));

        //scan through the spline's segments until we find the segment that contains b
        while(segmentLength < desiredLength)
        {
            index++;
            size_t wrappedIndex = index % spline.segmentCount();
            desiredLength -= segmentLength;
            segmentBegin = spline.segmentT(wrappedIndex);
            segmentLength = spline.segmentArcLength(wrappedIndex, segmentBegin, spline.segmentT(wrappedIndex + 1));
        }

        //index % segmentCount is the segment that contains b, now solve for b within this segment
        floating_t wrappedB = __ArcLengthSolvePrivate::solveSegment(spline, index % spline.segmentCount(), desiredLength, segmentLength, segmentBegin);

        //we now have to "unwrap" b
        floating_t initialWrap = a - wrappedA;
        size_t numCycles = index / spline.segmentCount();

        return wrappedB + initialWrap + numCycles * spline.getMaxT();
    }


    //subdivide the spline into pieces such that the arc length of each pieces is equal to desiredLength
    //returns a list of t values marking the boundaries of each piece
    //the first entry is always 0. the final entry is the T value that marks the end of the last cleanly-dividible piece
    //The remainder that could not be divided is the piece between the last entry and maxT
    template<template <class, typename> class Spline, class InterpolationType, typename floating_t>
    std::vector<floating_t> partition(const Spline<InterpolationType, floating_t>& spline, floating_t lengthPerPiece)
    {
        //first, compute total arc length and arc length for each segment
        std::vector<floating_t> segmentLengths(spline.segmentCount());
        floating_t totalArcLength(0);
        for(size_t i = 0; i < spline.segmentCount(); i++)
        {
            floating_t segmentLength = spline.segmentArcLength(i, spline.segmentT(i), spline.segmentT(i+1));
            totalArcLength += segmentLength;
            segmentLengths[i] = segmentLength;
        }

        size_t n = size_t(totalArcLength / lengthPerPiece) + 1;
        std::vector<floating_t> pieces(n);

        //set up the inter-piece state
        floating_t segmentRemainder = segmentLengths[0];
        size_t segmentIndex = 0;

        //for each piece, perform the same algorithm as the "solve" method, except re-use work between segments by referring to the segmentLengths array
        for(size_t i = 1; i < n; i++)
        {
            floating_t desiredLength = lengthPerPiece;
            floating_t segmentBegin = pieces[i - 1];

            //if the segment length is less than desiredLength, B will be in a different segment than A, so search though the spline until we find B's segment
            while(segmentRemainder < desiredLength)
            {
                segmentIndex++;
                desiredLength -= segmentRemainder;
                segmentRemainder = segmentLengths[segmentIndex];
                segmentBegin = spline.segmentT(segmentIndex);
            }

            //we've found the segment that b lies in, so solve for the remaining arc length within this segment
            pieces[i] = __ArcLengthSolvePrivate::solveSegment(spline, segmentIndex, desiredLength, segmentRemainder, segmentBegin);

            //set up the next iteration of the loop
            segmentRemainder = segmentRemainder - desiredLength;
        }
        return pieces;
    }

    //subdivide the spline into N pieces such that each piece has the same arc length
    //returns a list of N+1 T values, where return[i] is the T value of the beginning of a piece and return[i+1] is the T value of the end of a piece
    //the first element in the returned list is always 0, and the last element is always spline.getMaxT()
    template<template <class, typename> class Spline, class InterpolationType, typename floating_t>
    std::vector<floating_t> partitionN(const Spline<InterpolationType, floating_t>& spline, size_t n)
    {
        //first, compute total arc length and arc length for each segment
        std::vector<floating_t> segmentLengths(spline.segmentCount());
        floating_t totalArcLength(0);
        for(size_t i = 0; i < spline.segmentCount(); i++)
        {
            floating_t segmentLength = spline.segmentArcLength(i, spline.segmentT(i), spline.segmentT(i+1));
            totalArcLength += segmentLength;
            segmentLengths[i] = segmentLength;
        }
        const floating_t lengthPerPiece = totalArcLength / n;

        //set up the result vector
        std::vector<floating_t> pieces(n + 1);

        //set up the inter-piece state
        floating_t segmentRemainder = segmentLengths[0];
        size_t segmentIndex = 0;

        //for each piece, perform the same algorithm as the "solve" method, except re-use work between segments by referring to the segmentLengths array
        for(size_t i = 1; i < n; i++)
        {
            floating_t desiredLength = lengthPerPiece;
            floating_t segmentBegin = pieces[i - 1];

            //if the segment length is less than desiredLength, B will be in a different segment than A, so search though the spline until we find B's segment
            while(segmentRemainder < desiredLength)
            {
                segmentIndex++;
                desiredLength -= segmentRemainder;
                segmentRemainder = segmentLengths[segmentIndex];
                segmentBegin = spline.segmentT(segmentIndex);
            }

            //we've found the segment that b lies in, so solve for the remaining arc length within this segment
            pieces[i] = __ArcLengthSolvePrivate::solveSegment(spline, segmentIndex, desiredLength, segmentRemainder, segmentBegin);

            //set up the next iteration of the loop
            segmentRemainder = segmentRemainder - desiredLength;
        }

        pieces[n] = spline.getMaxT();
        return pieces;
    }
}
