/*
 * File: DLoopDetector.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: Generic include file for the DLoopDetector classes and
 *   the specialized loop detectors
 * License: see the LICENSE.txt file
 *
 */

/*! \mainpage DLoopDetector Library
 *
 * DLoopDetector library for C++:
 * Loop detector for monocular image sequences based on bags of words
 *
 * Written by Dorian Galvez-Lopez,
 * University of Zaragoza
 * 
 * Check my website to obtain updates: http://webdiis.unizar.es/~dorian
 *
 * \section requirements Requirements
 * This library requires the DUtils, DUtilsCV, DVision, DBoW2 and OpenCV libraries,
 * as well as the boost::dynamic_bitset class.
 *
 * \section citation Citation
 * If you use this software in academic works, please cite:
 <pre>
   @@ARTICLE{GalvezTRO12,
    author={Galvez-Lopez, Dorian and Tardos, J. D.}, 
    journal={IEEE Transactions on Robotics},
    title={Bags of Binary Words for Fast Place Recognition in Image Sequences},
    year={2012},
    month={October},
    volume={28},
    number={5},
    pages={1188--1197},
    doi={10.1109/TRO.2012.2197158},
    ISSN={1552-3098}
  }
 </pre>
 *
 */


#ifndef __D_T_LOOP_DETECTOR__
#define __D_T_LOOP_DETECTOR__

/// Loop detector for sequences of monocular images
namespace DLoopDetector
{
}

#include "DBoW2/DBoW2.h"
#include "TemplatedLoopDetector.h"
#include "DBoW2/FORB.h"
#include "DBoW2/FBrief.h"

/// SURF64 Loop Detector
typedef DLoopDetector::TemplatedLoopDetector
  <FORB::TDescriptor, FORB> OrbLoopDetector;

/// BRIEF Loop Detector
typedef DLoopDetector::TemplatedLoopDetector
  <FBrief::TDescriptor, FBrief> BriefLoopDetector;

#endif

