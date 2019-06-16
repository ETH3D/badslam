#pragma once

#include <vector>

#include "utils/spline_common.h"
#include "utils/calculus.h"

template<class InterpolationType, typename floating_t=float>
class Spline
{
public:
    Spline(std::vector<InterpolationType> originalPoints, floating_t maxT)
        :maxT(maxT), originalPoints(std::move(originalPoints))
    {}

public:
    struct InterpolatedPT;

    struct InterpolatedPTC;

    struct InterpolatedPTCW;

    virtual InterpolationType getPosition(floating_t x) const = 0;
    virtual InterpolatedPT getTangent(floating_t x) const = 0;
    virtual InterpolatedPTC getCurvature(floating_t x) const = 0;
    virtual InterpolatedPTCW getWiggle(floating_t x) const = 0;

    virtual floating_t arcLength(floating_t a, floating_t b) const = 0;
    virtual floating_t totalLength(void) const = 0;
    inline floating_t getMaxT(void) const { return maxT; }

    const std::vector<InterpolationType> &getOriginalPoints(void) const { return originalPoints; }
    virtual bool isLooping(void) const = 0;

    //lower level functions
    virtual size_t segmentCount(void) const = 0;
    virtual size_t segmentForT(floating_t t) const = 0;
    virtual floating_t segmentT(size_t segmentIndex) const = 0;
    virtual floating_t segmentArcLength(size_t segmentIndex, floating_t a, floating_t b) const = 0;

protected:
    const floating_t maxT;

private:
    const std::vector<InterpolationType> originalPoints;
};

template<class InterpolationType, typename floating_t=float>
class LoopingSpline: public Spline<InterpolationType, floating_t>
{
public:
    LoopingSpline(std::vector<InterpolationType> originalPoints, floating_t maxT)
        :Spline<InterpolationType, floating_t>(std::move(originalPoints), maxT)
    {}

    inline floating_t wrapT(floating_t t) const {
        float wrappedT = std::fmod(t, this->maxT);
        if(wrappedT < 0)
            return wrappedT + this->maxT;
        else
            return wrappedT;
    }
    virtual floating_t cyclicArcLength(floating_t a, floating_t b) const = 0;
};




template<template<class, typename> class SplineCore, class InterpolationType, typename floating_t>
class SplineImpl: public Spline<InterpolationType, floating_t>
{
public:
    InterpolationType getPosition(floating_t t) const override { return common.getPosition(t); }
    typename Spline<InterpolationType,floating_t>::InterpolatedPT getTangent(floating_t t) const override { return common.getTangent(t); }
    typename Spline<InterpolationType,floating_t>::InterpolatedPTC getCurvature(floating_t t) const override { return common.getCurvature(t); }
    typename Spline<InterpolationType,floating_t>::InterpolatedPTCW getWiggle(floating_t t) const override { return common.getWiggle(t); }

    floating_t arcLength(floating_t a, floating_t b) const override;
    floating_t totalLength(void) const override;

    bool isLooping(void) const override { return false; }

    size_t segmentCount(void) const override { return common.segmentCount(); }
    size_t segmentForT(floating_t t) const override { return common.segmentForT(t); }
    floating_t segmentT(size_t segmentIndex) const override { return common.segmentT(segmentIndex); }
    floating_t segmentArcLength(size_t segmentIndex, floating_t a, floating_t b) const override { return common.segmentLength(segmentIndex, a, b); }

protected:
    //protected constructor and destructor, so that this class can only be used as a parent class, even though it won't have any pure virtual methods
    SplineImpl(std::vector<InterpolationType> originalPoints, floating_t maxT)
        :Spline<InterpolationType, floating_t>(std::move(originalPoints), maxT)
    {}
    ~SplineImpl(void) = default;

    SplineCore<InterpolationType, floating_t> common;
};



template<template<class, typename> class SplineCore, class InterpolationType, typename floating_t>
class SplineLoopingImpl: public LoopingSpline<InterpolationType, floating_t>
{
public:
    InterpolationType getPosition(floating_t globalT) const override { return common.getPosition(this->wrapT(globalT)); }
    typename Spline<InterpolationType,floating_t>::InterpolatedPT getTangent(floating_t globalT) const override { return common.getTangent(this->wrapT(globalT)); }
    typename Spline<InterpolationType,floating_t>::InterpolatedPTC getCurvature(floating_t globalT) const override { return common.getCurvature(this->wrapT(globalT)); }
    typename Spline<InterpolationType,floating_t>::InterpolatedPTCW getWiggle(floating_t globalT) const override { return common.getWiggle(this->wrapT(globalT)); }

    floating_t arcLength(floating_t a, floating_t b) const override;
    floating_t cyclicArcLength(floating_t a, floating_t b) const override;
    floating_t totalLength(void) const override;

    bool isLooping(void) const override { return true; }

    size_t segmentCount(void) const override { return common.segmentCount(); }
    size_t segmentForT(floating_t t) const override { return common.segmentForT(this->wrapT(t)); }
    floating_t segmentT(size_t segmentIndex) const override { return common.segmentT(segmentIndex); }
    floating_t segmentArcLength(size_t segmentIndex, floating_t a, floating_t b) const override { return common.segmentLength(segmentIndex, a, b); }

protected:
    //protected constructor and destructor, so that this class can only be used as a parent class, even though it won't have any pure virtual methods
    SplineLoopingImpl(std::vector<InterpolationType> originalPoints, floating_t maxT)
        :LoopingSpline<InterpolationType, floating_t>(std::move(originalPoints), maxT)
    {}
    ~SplineLoopingImpl(void) = default;

    SplineCore<InterpolationType, floating_t> common;
};





template<class InterpolationType, typename floating_t>
struct Spline<InterpolationType,floating_t>::InterpolatedPT
{
    InterpolationType position;
    InterpolationType tangent;

    InterpolatedPT(const InterpolationType &p, const InterpolationType &t)
        :position(p),tangent(t)
    {}
};

template<class InterpolationType, typename floating_t>
struct Spline<InterpolationType,floating_t>::InterpolatedPTC
{
    InterpolationType position;
    InterpolationType tangent;
    InterpolationType curvature;

    InterpolatedPTC(const InterpolationType &p, const InterpolationType &t, const InterpolationType &c)
        :position(p),tangent(t),curvature(c)
    {}
};

template<class InterpolationType, typename floating_t>
struct Spline<InterpolationType,floating_t>::InterpolatedPTCW
{
    InterpolationType position;
    InterpolationType tangent;
    InterpolationType curvature;
    InterpolationType wiggle;

    InterpolatedPTCW(const InterpolationType &p, const InterpolationType &t, const InterpolationType &c, const InterpolationType &w)
        :position(p),tangent(t),curvature(c), wiggle(w)
    {}
};

template<template<class, typename> class SplineCore, class InterpolationType, typename floating_t>
floating_t SplineImpl<SplineCore, InterpolationType, floating_t>::arcLength(floating_t a, floating_t b) const
{
    if(a > b) {
        std::swap(a,b);
    }

    //get the knot indices for the beginning and end
    size_t aIndex = common.segmentForT(a);
    size_t bIndex = common.segmentForT(b);

    //if a and b occur inside the same segment, compute the length within that segment
    //but excude cases where a > b, because that means we need to wrap around
    if(aIndex == bIndex) {
        return common.segmentLength(aIndex, a, b);
    }
    else {
        //a and b occur in different segments, so compute one length for every segment
        floating_t result{0};

        //first segment
        floating_t aEnd = common.segmentT(aIndex + 1);
        result += common.segmentLength(aIndex, a, aEnd);

        //middle segments
        for(size_t i = aIndex + 1; i < bIndex; i++) {
            result += common.segmentLength(i, common.segmentT(i), common.segmentT(i + 1));
        }

        //last segment
        floating_t bBegin = common.segmentT(bIndex);
        result += common.segmentLength(bIndex, bBegin, b);

        return result;
    }
}

template<template<class, typename> class SplineCore, class InterpolationType, typename floating_t>
floating_t SplineImpl<SplineCore, InterpolationType, floating_t>::totalLength(void) const
{
    floating_t result{0};
    for(size_t i = 0; i < common.segmentCount(); i++) {
        result += common.segmentLength(i, common.segmentT(i), common.segmentT(i+1));
    }
    return result;
}


template<template<class, typename> class SplineCore, class InterpolationType, typename floating_t>
floating_t SplineLoopingImpl<SplineCore, InterpolationType, floating_t>::arcLength(floating_t a, floating_t b) const
{
    a = this->wrapT(a);
    b = this->wrapT(b);

    if(a > b) {
        std::swap(a,b);
    }

    //get the knot indices for the beginning and end
    size_t aIndex = common.segmentForT(a);
    size_t bIndex = common.segmentForT(b);

    //if a and b occur inside the same segment, compute the length within that segment
    //but excude cases where a > b, because that means we need to wrap around
    if(aIndex == bIndex) {
        return common.segmentLength(aIndex, a, b);
    }
    else {
        //a and b occur in different segments, so compute one length for every segment
        floating_t result{0};

        //first segment
        floating_t aEnd = common.segmentT(aIndex + 1);
        result += common.segmentLength(aIndex, a, aEnd);

        //middle segments
        for(size_t i = aIndex + 1; i < bIndex; i++) {
            result += common.segmentLength(i, common.segmentT(i), common.segmentT(i + 1));
        }

        //last segment
        floating_t bBegin = common.segmentT(bIndex);
        result += common.segmentLength(bIndex, bBegin, b);

        return result;
    }
}

//compute the arc length from a to b on the given spline, using wrapping/cyclic logic
//for cyclic splines only!
template<template <class, typename> class CyclicSplineT, class InterpolationType, typename floating_t>
floating_t SplineLoopingImpl<CyclicSplineT, InterpolationType, floating_t>::cyclicArcLength(floating_t a, floating_t b) const
{
    floating_t wrappedA = this->wrapT(a);
    floating_t wrappedB = this->wrapT(b);

    //if wrapped A is less than wrapped B, then we can use the normal arc legth formula
    if(wrappedA <= wrappedB)
    {
        return arcLength(wrappedA, wrappedB);
    }
    else
    {
        //get the knot indices for the beginning and end
        size_t aIndex = common.segmentForT(wrappedA);
        size_t bIndex = common.segmentForT(wrappedB);

        floating_t result{0};

        //first segment
        floating_t aEnd = common.segmentT(aIndex + 1);
        result += common.segmentLength(aIndex, wrappedA, aEnd);

        //for the "middle" segments. we're going to wrap around -- go from the segment after a to the end, then go from 0 to the segment before b
        for(size_t i = aIndex + 1; i < common.segmentCount(); i++) {
            result += common.segmentLength(i, common.segmentT(i), common.segmentT(i + 1));
        }

        //special case: if "b" is a multiple of maxT, then wrappedB wil be 0 and we don't need to bother computing the segments from T=0 to T=wrappedB
        if(wrappedB > 0)
        {
            for(size_t i = 0; i < bIndex; i++) {
                result += common.segmentLength(i, common.segmentT(i), common.segmentT(i + 1));
            }

            //last segment. if wrappedB == 0 then we've got a special case where b is maxT and was wrapped to 0, so we shouldn't compute the segment
            floating_t bBegin = common.segmentT(bIndex);
            result += common.segmentLength(bIndex, bBegin, wrappedB);
        }

        return result;
    }
}

template<template<class, typename> class SplineCore, class InterpolationType, typename floating_t>
floating_t SplineLoopingImpl<SplineCore, InterpolationType, floating_t>::totalLength(void) const
{
    floating_t result{0};
    for(size_t i = 0; i < common.segmentCount(); i++) {
        result += common.segmentLength(i, common.segmentT(i), common.segmentT(i+1));
    }
    return result;
}
