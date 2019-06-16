#pragma once

/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2011 Jose Luis Blanco (joseluisblancoc@gmail.com).
 *   All rights reserved.
 * Copyright 2014 Elliott Mahler (join.together@gmail.com).
 *   All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#include "nanoflann.hpp"
#include <vector>
#include <array>

template<int dimension, typename floating_t>
struct SplineSamples
{
    typedef floating_t coord_t; //!< The type of each coordinate

    struct Point
    {
        std::array<coord_t, dimension> coords;
        coord_t t;

        Point(const std::array<coord_t, dimension> &coords, coord_t ct)
            :coords(coords), t(ct)
        {}
    };

    std::vector<Point> pts;
};


template <typename Derived, int dimension>
struct SplineSampleAdaptor
{
    typedef typename Derived::coord_t coord_t;

    const Derived obj;

    /// The constructor that sets the data set source
    SplineSampleAdaptor(const Derived &obj_) : obj(obj_) {}

    /// CRTP helper method
    inline const Derived& derived() const { return obj; }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return derived().pts.size(); }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    inline coord_t kdtree_distance(const coord_t *p1, const size_t idx_p2,size_t size) const
    {
        coord_t sum = 0;
        for(size_t i = 0; i < size; i++) {
            coord_t diff = p1[i]-derived().pts[idx_p2].coords[i];
            sum += diff*diff;
        }
        return sum;
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline coord_t kdtree_get_pt(const size_t idx, int dim) const
    {
        return derived().pts[idx].coords[dim];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    // bb parameter is commented out because it's unused! we get obnoxious compiler warnings if we leave it uncommented
    template <class BBOX>
    bool kdtree_get_bbox(BBOX &/*bb*/) const { return false; }
};

template<int dimension, typename floating_t>
class SplineSampleTree
{
    typedef SplineSampleAdaptor<SplineSamples<dimension, floating_t>, dimension> AdaptorType;
    typedef nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<floating_t, AdaptorType>,
            AdaptorType, dimension>
        TreeType;

public:
    SplineSampleTree(const SplineSamples<dimension, floating_t> &samples)
        :adaptor(samples), tree(dimension, adaptor)
    {
        tree.buildIndex();
    }

    floating_t findClosestSample(const std::array<floating_t, dimension> &queryPoint) const
    {
        // do a knn search
        const size_t num_results = 1;
        size_t ret_index;
        floating_t out_dist_sqr;
        nanoflann::KNNResultSet<floating_t> resultSet(num_results);
        resultSet.init(&ret_index, &out_dist_sqr );
        tree.findNeighbors(resultSet, queryPoint.data(), nanoflann::SearchParams());

        return adaptor.derived().pts.at(ret_index).t;
    }

private:
    AdaptorType adaptor;
    TreeType tree;
};
