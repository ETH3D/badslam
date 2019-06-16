#pragma once

#include <vector>

class LinearAlgebra
{
private:
    LinearAlgebra() = default;

public:

    //solve the given tridiagonal matrix system, with the assumption that the lower diagonal and upper diagonal (ie secondaryDiagonal) are identical.
    //in other words, assume that the matrix is symmetric
    template<class OutputType, typename floating_t>
    static std::vector<OutputType> solveSymmetricTridiagonal(

            std::vector<floating_t> mainDiagonal,
            const std::vector<floating_t> secondaryDiagonal,
            std::vector<OutputType> inputVector);

    //solve the given tridiagonal matrix system
    template<class OutputType, typename floating_t>
    static std::vector<OutputType> solveTridiagonal(

            std::vector<floating_t> mainDiagonal,
            const std::vector<floating_t> upperDiagonal,
            const std::vector<floating_t> lowerDiagonal,
            std::vector<OutputType> inputVector);

    //solve the given cyclic tridiagonal matrix system, with the assumption that the lower diagonal and upper diagonal (ie secondaryDiagonal) are identical
    //in other words, assume that the matrix is symmetric
    template<class OutputType, typename floating_t>
    static std::vector<OutputType> solveCyclicSymmetricTridiagonal(

            std::vector<floating_t> mainDiagonal,
            std::vector<floating_t> secondaryDiagonal,
            std::vector<OutputType> inputVector);
};

template<class OutputType, typename floating_t>
std::vector<OutputType> LinearAlgebra::solveTridiagonal(

        std::vector<floating_t> mainDiagonal,
        const std::vector<floating_t> upperDiagonal,
        const std::vector<floating_t> lowerDiagonal,
        std::vector<OutputType> inputVector)
{
    //use the thomas algorithm to solve the tridiagonal matrix
    // http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

    //forward sweep
    for(size_t i = 1; i < inputVector.size(); i++)
    {
        floating_t m = lowerDiagonal[i - 1] / mainDiagonal[i - 1];
        mainDiagonal[i] -= m * upperDiagonal[i - 1];
        inputVector[i] -= m * inputVector[i - 1];
    }

    //back substitution
    size_t finalIndex = inputVector.size();
    inputVector[finalIndex - 1] /= mainDiagonal[finalIndex - 1];

    for(size_t i = finalIndex - 1; i > 0; i--)
    {
        inputVector[i - 1] = (inputVector[i - 1] - upperDiagonal[i - 1] * inputVector[i]) / mainDiagonal[i - 1];
    }

    return inputVector;
}

template<class OutputType, typename floating_t>
std::vector<OutputType> LinearAlgebra::solveSymmetricTridiagonal(

        std::vector<floating_t> mainDiagonal,
        const std::vector<floating_t> secondaryDiagonal,
        std::vector<OutputType> inputVector)
{
    //use the thomas algorithm to solve the tridiagonal matrix
    // http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

    //forward sweep
    for(size_t i = 1; i < inputVector.size(); i++)
    {
        floating_t m = secondaryDiagonal[i - 1] / mainDiagonal[i - 1];
        mainDiagonal[i] -= m * secondaryDiagonal[i - 1];
        inputVector[i] -= m * inputVector[i - 1];
    }

    //back substitution
    size_t finalIndex = inputVector.size();
    inputVector[finalIndex - 1] /= mainDiagonal[finalIndex - 1];

    for(size_t i = finalIndex - 1; i > 0; i--)
    {
        inputVector[i - 1] = (inputVector[i - 1] - secondaryDiagonal[i - 1] * inputVector[i]) / mainDiagonal[i - 1];
    }

    return inputVector;
}

template<class OutputType, typename floating_t>
std::vector<OutputType> LinearAlgebra::solveCyclicSymmetricTridiagonal(

        std::vector<floating_t> mainDiagonal,
        std::vector<floating_t> secondaryDiagonal,
        std::vector<OutputType> inputVector)
{
    //apply the sherman-morrison algorithm to the cyclic tridiagonal matrix so that we can use the standard tridiagonal algorithm
    //we're getting this algorithm from http://www.cs.princeton.edu/courses/archive/fall11/cos323/notes/cos323_f11_lecture06_linsys2.pdf
    //basically, we're going to solve two different non-cyclic versions of this system and then combine the results

    size_t size = inputVector.size();


    //the value at the upper right and lower left of the input matrix. it's at the end of the secondary diagonal array because almost all
    //cyclic tridiagonal papers treat it as an extension of the secondary diagonals
    floating_t cornerValue = secondaryDiagonal.at(size - 1);

    //gamma value - doesn't affect actual output (the algorithm makes sure it cancels out), but a good choice for this value can reduce floating point errors
    floating_t gamma = -mainDiagonal.at(0);
    floating_t cornerMultiplier = cornerValue/gamma;

    //corrective vector U: should be all 0, except for gamma in the first element, and cornerValue at the end
    std::vector<floating_t> correctionInputU(size);
    correctionInputU[0] = gamma;
    correctionInputU[size - 1] = cornerValue;

    //modify the main diagonal of the matrix to account for the correction vector
    mainDiagonal[0] -= gamma;
    mainDiagonal[size - 1] -= cornerValue * cornerMultiplier;

    //solve the modified system for the input vector
    std::vector<OutputType> initialOutput = solveSymmetricTridiagonal(
                mainDiagonal,
                secondaryDiagonal,
                std::move(inputVector)
                );

    //solve the modified system for the correction vector
    std::vector<floating_t> correctionOutput = solveSymmetricTridiagonal(
                std::move(mainDiagonal),
                std::move(secondaryDiagonal),
                std::move(correctionInputU)
                );

    //compute the corrective OutputType to apply to each initial output
    //this involves a couple dot products, but all of the elements on the correctionV vector are 0 except the first and last
    //so just compute those directly instead of looping through and multplying a bunch of 0s
    OutputType factor = (initialOutput.at(0) + initialOutput.at(size - 1) * cornerMultiplier) / (1 + correctionOutput.at(0) + correctionOutput.at(size - 1) * cornerMultiplier);

    /*std::vector<floating_t> correctionV(size);
    correctionV[0] = 1;
    correctionV[size - 1] = cornerMultiplier;
    OutputType factor = vectorDotProduct(initialOutput, correctionV) / (1 + vectorDotProduct(correctionV, correctionOutput));*/

    //use the correction factor to modify the result
    for(size_t i = 0; i < size; i++)
    {
        initialOutput[i] -= factor * correctionOutput.at(i);
    }

    return initialOutput;
}
