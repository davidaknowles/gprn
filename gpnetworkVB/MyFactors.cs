using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Distributions.Kernels;

namespace gpnetworkVB
{
    /// <summary>
    /// Provides two custom factors
    /// </summary>
    public static class MyFactors
    {
        /// <summary>
        /// A zero mean multivariate Gaussian with a scaling parameter that can be stochastic
        /// </summary>
        [Stochastic]
        [ParameterNames("Sample", "scaling", "precision")]
        public static Vector VectorGaussianScaled(double scaling, PositiveDefiniteMatrix precision)
        {
            var scaledPrec = new PositiveDefiniteMatrix(precision);
            scaledPrec.Scale(scaling);
            return VectorGaussian.FromMeanAndPrecision(Vector.Zero(precision.Cols), scaledPrec).Sample();
        }

        /// <summary>
        /// A Gaussian process regression with specified kernel and the possibility of choosing which
        /// hypers should be optimised
        /// </summary>
        [Stochastic]
        [ParameterNames("Sample", "scaling", "x", "hypersToOptimise", "initialKernel")]
        public static Vector GP(double scaling, Vector[] x, int[] hypersToOptimise, KernelFunction initialKernel)
        {
            return VectorGaussian.Uniform(x.Length).Sample();
        }
    }
}
