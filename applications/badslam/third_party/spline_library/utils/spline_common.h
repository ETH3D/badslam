#pragma once

#include <unordered_map>
#include <vector>
#include <cmath>

namespace SplineCommon
{
    //compute the T values for the given points, with the given alpha.
    //the distance in T between adjacent points is the magitude of the distance, raised to the power alpha
    template<class InterpolationType, typename floating_t>
    floating_t computeTDiff(InterpolationType p1, InterpolationType p2, floating_t alpha);


    //compute t values for the given points, based on the alpha value
    //if innerPadding > 0, the first 'innerPadding-1' values will be negative, and the innerPadding'th value will be 0
    //so the spline will effectively begin at innerPadding + 1
    //this is used for splines like catmull-rom, where the first and last point are used ONLY to calculate tangent
    template<class InterpolationType, typename floating_t>
    std::vector<floating_t> computeTValuesWithInnerPadding(
            const std::vector<InterpolationType> &points,
            floating_t alpha,
            size_t innerPadding
            );

    //compute the T values for the given points, with the given alpha, for use in a looping spline
    //if padding is zero, this method will return points.size() + 1 points
    //the "extra" point is because the first point in the list is represented at the beginning AND end
    //if padding is > 0, this method will also compute "extra" T values before the beginning and after the end
    //these won't actually add any extra information, but help simplify calculations that wrap around the loop
    template<class InterpolationType, typename floating_t>
    std::vector<floating_t> computeLoopingTValues(const std::vector<InterpolationType> &points, floating_t alpha, size_t padding);



    //given a list of knots and a t value, return the index of the knot the t value falls within
    template<typename floating_t>
    size_t getIndexForT(const std::vector<floating_t> &knotData, floating_t t);
}

template<class InterpolationType, typename floating_t>
floating_t SplineCommon::computeTDiff(InterpolationType p1, InterpolationType p2, floating_t alpha)
{
    if(alpha == 0)
    {
        return 1;
    }
    else
    {
        auto distanceSq = (p1 - p2).lengthSquared();

        //if these points are right on top of each other, don't bother with the power calculation
        if(distanceSq < .0001)
        {
            return 0;
        }
        else
        {
            //multiply alpha by 0.5 so that we tke the square root of distanceSq
            //ie: result = distance ^ alpha, and distance = (distanceSq)^(0.5)
            //so: result = (distanceSq^0.5)^(alpha) = (distanceSq)^(0.5*alpha)
            //this way we don't have to do a pow AND a sqrt
            return pow(distanceSq, alpha * 0.5);
        }
    }
}

template<class InterpolationType, typename floating_t>
std::vector<floating_t> SplineCommon::computeTValuesWithInnerPadding(
        const std::vector<InterpolationType> &points,
        floating_t alpha,
        size_t innerPadding
        )
{
    size_t size = points.size();
    size_t endPaddingIndex = size - 1 - innerPadding;
    size_t desiredMaxT = size - 2 * innerPadding - 1;

    std::vector<floating_t> tValues(size);

    //we know points[padding] will have a t value of 0
    tValues[innerPadding] = 0;

    //loop backwards from padding to give the earlier points negative t values
    for(size_t i = innerPadding; i > 0; i--)
    {
        //Points inside the padding will not be interpolated
        //so give it a negative t value, so that the first actual point can have a t value of 0
        tValues[i - 1] = tValues[i] - computeTDiff(points[i - 1], points[i], alpha);
    }

    //compute the t values of the other points
    for(size_t i = 1; i < size - innerPadding; i++)
    {
        tValues[i + innerPadding] = tValues[i - 1 + innerPadding] + computeTDiff(points[i], points[i - 1], alpha);
    }

    //we want to know the t value of the last segment so that we can normalize them all
    floating_t maxTRaw = tValues[endPaddingIndex];

    //now that we have all ouf our t values and indexes figured out, normalize the t values by dividing them by maxT
    floating_t multiplier = desiredMaxT / maxTRaw;
    for(auto &entry: tValues)
    {
        entry *= multiplier;
    }

    return tValues;
}

template<class InterpolationType, typename floating_t>
std::vector<floating_t> SplineCommon::computeLoopingTValues(
        const std::vector<InterpolationType> &points,
        floating_t alpha,
        size_t padding)
{
    size_t size = points.size();
    floating_t maxT = floating_t(size);
    std::vector<floating_t> tValues(size + padding * 2 + 1);

    //compute the t values each point
    tValues[padding] = 0;
    for(size_t i = 1; i < size; i++)
    {
        tValues[i + padding] = tValues[i - 1 + padding] + computeTDiff(points[i], points[i - 1], alpha);
    }

    //the final t value wraps around to the beginning
    tValues[size + padding] = tValues[size - 1 + padding] + computeTDiff(points[size - 1], points[0], alpha);

    //we want to know the t value of the last segment so that we can normalize them all
    floating_t maxTRaw = tValues[size + padding];

    //now that we have all ouf our t values and indexes figured out, normalize the t values by dividing them by maxT
    floating_t multiplier = maxT / maxTRaw;
    for(size_t i = 1; i < size + 1; i++) {
       tValues[i + padding] *= multiplier;
    }

    //add padding in addition to the points - 2 on each end
    //we calculate the padding by basically wraping the difference in T values
    for(size_t i = 0; i < padding; i++)
    {
        tValues[i] = tValues[i + size] - maxT;
        tValues[i + size + padding + 1] = tValues[i + padding + 1] + maxT;
    }

    return tValues;
}


template<typename floating_t>
size_t SplineCommon::getIndexForT(const std::vector<floating_t> &knotData, floating_t t)
{
    //we want to find the segment whos t0 and t1 values bound x

    //if no segments bound x, return -1
    if(t <= knotData.front())
        return 0;
    if(t >= knotData.back())
        return knotData.size() - 1;

    //our initial guess will be to subtract the minimum t value, then take the floor
    size_t currentIndex = std::floor(t - knotData.front());
    size_t size = knotData.size();

    //move left or right in the array until we've found the correct index
    size_t searchSize = 1;
    while(t < knotData[currentIndex])
    {
        while(currentIndex >= 0 && t < knotData[currentIndex])
        {
            searchSize++;
            currentIndex -= searchSize;
        }
        if(currentIndex < 0 || t > knotData[currentIndex + 1])
        {
            currentIndex += searchSize;
            searchSize /= 4;
        }

    }
    while(t >= knotData[currentIndex + 1])
    {
        while(currentIndex < size && t >= knotData[currentIndex])
        {
            searchSize++;
            currentIndex += searchSize;
        }
        if(currentIndex >= size || t < knotData[currentIndex])
        {
            currentIndex -= searchSize;
            searchSize /= 4;
        }
    }
    return currentIndex;
}
