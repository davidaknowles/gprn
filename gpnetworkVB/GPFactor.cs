
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq; 
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Distributions.Kernels;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Utils;
using MicrosoftResearch.Infer.Models; 
using MicrosoftResearch.Infer;

namespace gpnetworkVB
{
    // Buffer class used by GPFactor
    [Serializable]
    public class GPBuffer
    {
        public PositiveDefiniteMatrix Precision;
        public KernelFunction kernel;
        public double ESamplePrecisionSample;
        public double PrecisionMeanLogDet;
    }

    /// <summary>
    /// This operator extends Infer.NET with a Gaussian Process regression factor capable of using EM to learn its 
    /// hyperparameters. Used inside an Infer.NET VMP algorithm this allows variational EM. The main limitation 
    /// is that hypers cannot be shared this way across multiple GPs. To get around this the current solution is
    /// to manually edit the Infer.NET compiled algorithm source code
    /// </summary>
    [FactorMethod(typeof(MyFactors), "GP")]
    [Buffers("SampleMean", "SampleVariance", "Buffer")]
    [Quality(QualityBand.Experimental)]
    public static class GPFactor
    {
        public static Settings settings;

        [Skip]
        public static GPBuffer BufferInit([IgnoreDependency] KernelFunction initialKernel, [IgnoreDependency] Vector[] x)
        {
            return new GPBuffer
            {
                Precision = Utils.GramMatrix(initialKernel, x).Inverse(),
                //kernel = ObjectCloner.Clone(initialKernel),
                kernel = initialKernel
            };
        }

        /// <summary>
        /// This is just an easy but non-general way to store what the learnt kernel was (to get round the
        /// fact that you currently can't query Infer.NET for the contents of a buffer)
        /// </summary>
        public static KernelFunction rememberKernel; 

        /// <summary>
        /// Uses the KernelOptimiser class to optimise the hypers given the current variational posterior 
        /// on the function values (which has mean SampleMean and covariance SampleCovariance)
        /// </summary>
        public static GPBuffer BufferHelper(int[] hypersToOptimise, GPBuffer Buffer, Vector[] x, Vector SampleMean, PositiveDefiniteMatrix SampleVariance, Gamma scaling)
        {
            if (SampleMean.All(o => o == 0.0))
            {
                Buffer.Precision = Utils.GramMatrix(Buffer.kernel, x).Inverse();
            }
            else
            {
                //Console.WriteLine(Utils.KernelToArray(Buffer.kernel).Select(o => o.ToString()).Aggregate((p, q) => p + " " + q));
                var helper = new KernelOptimiser(settings);
                helper.kernel = Buffer.kernel;
                helper.xData = x;
                helper.hypersToOptimise = hypersToOptimise;
                helper.Optimise((prec, gradK, gradientVector) =>
                     helperFunction(prec, gradK, gradientVector, scaling, SampleMean,
                     SampleVariance), ref Buffer.Precision);
                Buffer.ESamplePrecisionSample = VectorGaussianScaledPrecisionOp.ESamplePrecisionSample(SampleMean, SampleVariance, Buffer.Precision);
                Buffer.PrecisionMeanLogDet = VectorGaussianScaledPrecisionOp.PrecisionMeanLogDet(Buffer.Precision);
                //Console.WriteLine(Utils.KernelToArray(Buffer.kernel).Select(o => o.ToString()).Aggregate((p, q) => p + " " + q));
                rememberKernel = Buffer.kernel;
            }
            return Buffer;
        }

        public static GPBuffer Buffer(int[] hypersToOptimise, GPBuffer Buffer, Vector[] x, [SkipIfUniform] Vector SampleMean, [SkipIfUniform] PositiveDefiniteMatrix SampleVariance, [SkipIfUniform] Gamma scaling)
        {
            return BufferHelper(hypersToOptimise, Buffer, x, SampleMean, SampleVariance, scaling); 
        }

        public static GPBuffer Buffer(int[] hypersToOptimise, GPBuffer Buffer, Vector[] x, [SkipIfUniform] Vector SampleMean, [SkipIfUniform] PositiveDefiniteMatrix SampleVariance, double scaling)
        {
            return BufferHelper(hypersToOptimise, Buffer, x, SampleMean, SampleVariance, Gamma.PointMass(scaling)); 
        }

        /// <summary>
        /// Calculates the variational lower bound and its derivatives wrt to the hypers. gradK is the gradient of the kernel matrix
        /// wrt to the hypers. 
        /// </summary>
        public static double helperFunction(PositiveDefiniteMatrix prec, PositiveDefiniteMatrix[] gradK, Vector gradientVector, Gamma scaling,
            Vector functions_SampleMean,
           PositiveDefiniteMatrix functions_SampleVariance)
        {
            double res = 0;
            var logDetPrec = VectorGaussianScaledPrecisionOp.PrecisionMeanLogDet(prec);

            res = VectorGaussianScaledPrecisionOp.AverageLogFactor(functions_SampleMean, scaling,
                VectorGaussianScaledPrecisionOp.ESamplePrecisionSample(functions_SampleMean, functions_SampleVariance, prec),
                logDetPrec);
            if (gradientVector != null)
                for (int i = 0; i < gradientVector.Count; i++)
                {
                    gradientVector[i] = VectorGaussianScaledPrecisionOp.GradAverageLogFactor(
                        functions_SampleMean,
                        functions_SampleVariance,
                        Gamma.PointMass(1),
                        gradK[i],
                        prec);
                }
            return res;
        }

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
            GPBuffer Buffer)
        {
            return VectorGaussianScaledPrecisionOp.AverageLogFactor(SampleMean, scaling, Buffer.ESamplePrecisionSample, Buffer.PrecisionMeanLogDet); 
        }

        public static double AverageLogFactor(
            Vector SampleMean,
            double scaling,
            GPBuffer Buffer)
        {
            return VectorGaussianScaledPrecisionOp.AverageLogFactor(SampleMean, scaling, Buffer.ESamplePrecisionSample, Buffer.PrecisionMeanLogDet); 
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
        public static VectorGaussian SampleAverageLogarithm([SkipIfUniform] Gamma scaling, GPBuffer Buffer, VectorGaussian result)
        {
            result.Precision.SetTo(Buffer.Precision);
            result.Precision.Scale(scaling.GetMean());
            return result;
        }

        public static VectorGaussian SampleAverageLogarithm(double scaling, GPBuffer Buffer, VectorGaussian result)
        {
            result.Precision.SetTo(Buffer.Precision);
            if (scaling != 1.0)
                result.Precision.Scale(scaling);
            return result;
        }

        public static Gamma ScalingAverageLogarithm(Vector SampleMean, PositiveDefiniteMatrix SampleVariance,
            [Fresh, SkipIfUniform] GPBuffer Buffer)
        {
            return Gamma.FromNatural(.5 * Buffer.Precision.Rows, .5 * Buffer.ESamplePrecisionSample);
        }

        public static Gamma ScalingAverageLogarithm([Fresh] GPBuffer Buffer) //SampleVector SampleMean, [Fresh, SkipIfUniform] PositiveDefiniteMatrix SampleVariance, )
        {
            return Gamma.FromNatural(.5 * Buffer.Precision.Rows, .5 * Buffer.ESamplePrecisionSample/*(SampleMean, SampleVariance, Precision)*/);
        }


    }

}
