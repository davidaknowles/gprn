using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Distributions.Kernels;

namespace gpnetworkVB
{
    public static class Utilities
    {
        /// <summary>
        /// Just an OR operation on a 2D binary array
        /// </summary>
        /// <param name="isMissing"></param>
        /// <returns></returns>
        private static bool AnyIsMissing(bool[,] isMissing)
        {
            for (int d = 0; d < isMissing.GetLength(0); d++)
                for (int n = 0; n < isMissing.GetLength(1); n++)
                    if (isMissing[d, n])
                        return true;
            return false;
        }

        /// <summary>
        /// Assess the fitted model on held out data. 
        /// </summary>
        /// <param name="model">Fitted model</param>
        /// <param name="data">All data</param>
        /// <param name="isMissing">Which elements are missing (heldout)</param>
        /// <param name="logTransform">Whether the data was log transformed</param>
        /// <param name="means">The means used to normalise the data, if normalised</param>
        /// <param name="vars">The variances used to normalise the data, if normalised</param>
        /// <returns>Array of error measures</returns>
        public static double[] AssessOnMissing(INetworkModel model, double[,] data, bool[,] isMissing, bool logTransform, double[] means = null, double[] vars = null)
        {
            double logProb = 0, error = 0, counter = 0;
            for (int d = 0; d < data.GetLength(0); d++)
            {
                for (int n = 0; n < data.GetLength(1); n++)
                {
                    if (isMissing[d, n])
                    {
                        counter += 1;
                        var prediction = GaussianOp.SampleAverageLogarithm(model.Marginal<Gaussian[,]>("noiseLessY")[d, n], model.Marginal<Gamma[]>("noisePrecisionArray")[d]);
                        logProb += prediction.GetLogProb(data[d, n]);
                        double predMean = prediction.GetMean();
                        double dataPoint = data[d, n];
                        if (means != null)
                        {
                            predMean = predMean * Math.Sqrt(vars[d]) + means[d];
                            dataPoint = dataPoint * Math.Sqrt(vars[d]) + means[d];
                        }
                        if (logTransform)
                        {
                            predMean = Math.Exp(predMean);
                            dataPoint = Math.Exp(dataPoint);
                        }
                        error += Math.Abs(predMean - dataPoint);
                    }
                }
            }
            logProb /= counter;
            error /= counter;
            return new double[] { logProb, error };
        }


        /// <summary>
        /// Get the predictive distribution for the N+1 time, used for the multivariate volatility experiments
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        public static VectorGaussian CorrelatedPredictions(INetworkModel model)
        {
            var noisePrecisionPost = model.Marginal<Gamma[]>("noisePrecisionArray");
            var f = model.Marginal<Gaussian[][]>("nodeFunctionValuesPredictive");
            var W = model.Marginal<Gaussian[,][]>("weightFunctionValuesPredictive");
            int ni = model.N - 1;
            var mean = Vector.Zero(model.D);
            var cov = new PositiveDefiniteMatrix(model.D, model.D);
            for (int i = 0; i < model.D; i++)
            {
                cov[i, i] = noisePrecisionPost[i].GetMeanInverse();
                for (int k = 0; k < model.Q; k++)
                {
                    mean[i] += W[i, k][ni].GetMean() * f[k][ni].GetMean();
                    cov[i, i] += W[i, k][ni].GetVariance() * (f[k][ni].GetMean() * f[k][ni].GetMean() + f[k][ni].GetVariance());
                }
                for (int j = 0; j < model.D; j++)
                {
                    for (int k = 0; k < model.Q; k++)
                        cov[i, j] += W[i, k][ni].GetMean() * W[j, k][ni].GetMean() * f[k][ni].GetVariance();
                }
            }
            return VectorGaussian.FromMeanAndVariance(mean, cov);
        }

        /// <summary>
        /// Get the fitted covariance over the length of the time series
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        public static PositiveDefiniteMatrix[] HistoricalPredictiveCovariances(INetworkModel model)
        {
            var noisePrecisionPost = model.Marginal<Gamma[]>("noisePrecisionArray");
            var f = model.Marginal<Gaussian[][]>("nodeFunctionValuesPredictive");
            var W = model.Marginal<Gaussian[,][]>("weightFunctionValuesPredictive");
            var result = new PositiveDefiniteMatrix[model.N];
            for (int ni = 0; ni < model.N; ni++)
            {
                var mean = Vector.Zero(model.D);
                var cov = new PositiveDefiniteMatrix(model.D, model.D);
                for (int i = 0; i < model.D; i++)
                {
                    cov[i, i] = noisePrecisionPost[i].GetMeanInverse();
                    for (int k = 0; k < model.Q; k++)
                    {
                        mean[i] += W[i, k][ni].GetMean() * f[k][ni].GetMean();
                        cov[i, i] += W[i, k][ni].GetVariance() * (f[k][ni].GetMean() * f[k][ni].GetMean() + f[k][ni].GetVariance());
                    }
                    for (int j = 0; j < model.D; j++)
                    {
                        for (int k = 0; k < model.Q; k++)
                            cov[i, j] += W[i, k][ni].GetMean() * W[j, k][ni].GetMean() * f[k][ni].GetVariance();
                    }
                    result[ni] = cov;
                }
            }
            return result;
        }

        /// <summary>
        /// Get the fitted NOISE covariance only over the time series
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        public static PositiveDefiniteMatrix[] HistoricalNoiseCovariances(INetworkModel model)
        {
            var noisePrecisionPost = model.Marginal<Gamma[]>("noisePrecisionArray");
            var nodeSignalPrecisionsPost = model.Marginal<Gamma[]>("nodeSignalPrecisions");
            var W = model.Marginal<Gaussian[,][]>("weightFunctionValuesPredictive");
            var result = new PositiveDefiniteMatrix[model.N];
            for (int ni = 0; ni < model.N; ni++)
            {
                var cov = new PositiveDefiniteMatrix(model.D, model.D);
                for (int i = 0; i < model.D; i++)
                {
                    cov[i, i] = noisePrecisionPost[i].GetMeanInverse();
                    for (int j = 0; j < model.D; j++)
                    {
                        for (int k = 0; k < model.Q; k++)
                            cov[i, j] += W[i, k][ni].GetMean() * W[j, k][ni].GetMean() * nodeSignalPrecisionsPost[k].GetMeanInverse();
                    }
                }
                result[ni] = cov;
            }
            return result;
        }
    }
}
