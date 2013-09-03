// (C) Copyright 2008 Microsoft Research Cambridge
using System;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Utils;
using MicrosoftResearch.Infer; 

namespace gpnetworkVB
{
    /// <summary>
    /// Provides outgoing messages for <see cref="Factor.VectorGaussian"/>, given random arguments to the function.
    /// </summary>
    [FactorMethod(typeof(MyFactors), "VectorGaussianScaled")]
    [Buffers("SampleMean", "SampleVariance", "ESamplePrecisionSample", "PrecisionMeanLogDet")]
    [Quality(QualityBand.Experimental)]
    public static class VectorGaussianScaledPrecisionOp
    {
        /// <summary>
        /// Initialise the buffer 'SampleVariance'
        /// </summary>
        /// <param name="Sample">Incoming message from 'sample'.</param>
        /// <returns>Initial value of buffer 'SampleVariance'</returns>
        /// <remarks><para>
        /// 
        /// </para></remarks>
        [Skip]
        public static PositiveDefiniteMatrix SampleVarianceInit([IgnoreDependency] VectorGaussian Sample)
        {
            return new PositiveDefiniteMatrix(Sample.Dimension, Sample.Dimension);
        }
        /// <summary>
        /// Update the buffer 'SampleVariance'
        /// </summary>
        /// <param name="Sample">Incoming message from 'sample'. Must be a proper distribution.  If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message</param>
        /// <returns><paramref name="result"/></returns>
        /// <remarks><para>
        /// 
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="Sample"/> is not a proper distribution</exception>
        public static PositiveDefiniteMatrix SampleVariance([Proper] VectorGaussian Sample, PositiveDefiniteMatrix result)
        {
            return Sample.GetVariance(result);
        }
        
        /// <summary>
        /// Initialise the buffer 'SampleMean'
        /// </summary>
        /// <param name="Sample">Incoming message from 'sample'.</param>
        /// <returns>Initial value of buffer 'SampleMean'</returns>
        /// <remarks><para>
        /// 
        /// </para></remarks>
        [Skip]
        public static Vector SampleMeanInit([IgnoreDependency] VectorGaussian Sample)
        {
            return Vector.Zero(Sample.Dimension);
        }
        
        /// <summary>
        /// Update the buffer 'SampleMean'
        /// </summary>
        /// <param name="Sample">Incoming message from 'sample'. Must be a proper distribution.  If any element is uniform, the result will be uniform.</param>
        /// <param name="SampleVariance">Buffer 'SampleVariance'.</param>
        /// <param name="result">Modified to contain the outgoing message</param>
        /// <returns><paramref name="result"/></returns>
        /// <remarks><para>
        /// 
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="Sample"/> is not a proper distribution</exception>
        public static Vector SampleMean([Proper] VectorGaussian Sample, [Fresh] PositiveDefiniteMatrix SampleVariance, Vector result)
        {
            return Sample.GetMean(result, SampleVariance);
        }

        /// <summary>
        /// Update the buffer 'PrecisionMeanLogDet'
        /// </summary>
        /// <param name="Precision">Incoming message from 'precision'.</param>
        /// <returns>New value of buffer 'PrecisionMeanLogDet'</returns>
        /// <remarks><para>
        /// 
        /// </para></remarks>
        public static double PrecisionMeanLogDet([Proper] PositiveDefiniteMatrix Precision)
        {
            return Precision.LogDeterminant(); 
        }

        /// <summary>
        /// Evidence message for VMP
        /// </summary>
        /// <param name="sample">Incoming message from 'sample'. Must be a proper distribution.  If any element is uniform, the result will be uniform.</param>
        /// <param name="SampleMean">Buffer 'SampleMean'.</param>
        /// <param name="SampleVariance">Buffer 'SampleVariance'.</param>
        /// <param name="mean">Constant value for 'mean'.</param>
        /// <param name="precision">Constant value for 'precision'.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions</returns>
        /// <remarks><para>
        /// The formula for the result is <c>sum_(sample) p(sample) log(factor(sample,mean,precision))</c>.
        /// Adding up these values across all factors and variables gives the log-evidence estimate for VMP.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="sample"/> is not a proper distribution</exception>
        public static double AverageLogFactor(
            Vector SampleMean,
            Gamma scaling,
            double ESamplePrecisionSample,
            double PrecisionMeanLogDet)
        {
            int dim = SampleMean.Count;
            return -dim * MMath.LnSqrt2PI + .5 * dim * scaling.GetMeanLog() + 0.5 * (PrecisionMeanLogDet 
                 - scaling.GetMean() * ESamplePrecisionSample );
        }

        public static double GradAverageLogFactor(
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            Gamma scaling,
            PositiveDefiniteMatrix gradK,
            PositiveDefiniteMatrix Precision)
        {
            int dim = SampleMean.Count;
            // TODO: this is currently O(n^3) but can be O(n^2)
            double GradESamplePrecisionSample = ESamplePrecisionSample(SampleMean, SampleVariance, 
                new PositiveDefiniteMatrix((Matrix)Precision * (Matrix)gradK * (Matrix)Precision));
            double GradPrecisionMeanLogDet = 0.0;
            for (int i = 0; i < SampleMean.Count; i++)
            {
                for (int l = 0; l < SampleMean.Count; l++)
                {
                    GradPrecisionMeanLogDet += Precision[i, l] * gradK[l, i];
                }
            }
            return 0.5 * (-GradPrecisionMeanLogDet +   scaling.GetMean() * GradESamplePrecisionSample);
        }

        public static double AverageLogFactor(
            Vector SampleMean,
            double scaling,
            double ESamplePrecisionSample,
            double PrecisionMeanLogDet)
        {
            int dim = SampleMean.Count;
            return -dim * MMath.LnSqrt2PI + .5 * dim * Math.Log(scaling) + 0.5 * (PrecisionMeanLogDet - scaling * ESamplePrecisionSample);
        }

        public static double ESamplePrecisionSample(
            [Fresh] Vector SampleMean,
            [Fresh] PositiveDefiniteMatrix SampleVariance,
            PositiveDefiniteMatrix precision)
        {
            int dim = SampleMean.Count;
            double precTimesVariance = 0.0;
            double precTimesDiff = 0.0;
            for (int i = 0; i < dim; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < dim; j++)
                {
                    sum += precision[i, j] * SampleMean[j];
                    precTimesVariance += precision[i, j] * SampleVariance[i, j];
                }
                precTimesDiff += sum * SampleMean[i];
            }
            return precTimesDiff + precTimesVariance;
        }

        /// <summary>
        /// VMP message to 'sample'
        /// </summary>
        /// <param name="Mean">Constant value for 'mean'.</param>
        /// <param name="Precision">Constant value for 'precision'.</param>
        /// <param name="result">Modified to contain the outgoing message</param>
        /// <returns><paramref name="result"/></returns>
        /// <remarks><para>
        /// The outgoing message is the factor viewed as a function of 'sample' conditioned on the given values.
        /// </para></remarks>
        public static VectorGaussian SampleAverageLogarithm([SkipIfUniform] Gamma scaling, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            result.Precision.SetTo(Precision);
            result.Precision.Scale(scaling.GetMean());
            return result;
        }

        public static VectorGaussian SampleAverageLogarithm(double scaling, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            result.Precision.SetTo(Precision);
            if (scaling != 1.0)
                result.Precision.Scale(scaling);
            return result;
        }

        public static Gamma ScalingAverageLogarithm(PositiveDefiniteMatrix Precision, Vector SampleMean, PositiveDefiniteMatrix SampleVariance)
        {
            return Gamma.FromNatural(.5 * Precision.Rows, .5 * ESamplePrecisionSample(SampleMean, SampleVariance, Precision)); 
        }

        public static Gamma ScalingAverageLogarithm(PositiveDefiniteMatrix Precision, [Fresh, SkipIfUniform] double ESamplePrecisionSample) //SampleVector SampleMean, [Fresh, SkipIfUniform] PositiveDefiniteMatrix SampleVariance, )
        {
            return Gamma.FromNatural(.5 * Precision.Rows, .5 * ESamplePrecisionSample/*(SampleMean, SampleVariance, Precision)*/);
        }


    }

}
