using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Distributions.Kernels;
using MicrosoftResearch.Infer.Distributions;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Reflection;

namespace gpnetworkVB
{
    /// <summary>
    /// Replicates the parallel for loop from .NET 4.0 using .NET 3.0 functionality. 
    /// </summary>
    public class MyParallel
    {

        public delegate void Method(int i);

        /// <summary>
        /// Executes a set of methods in parallel and returns the results
        /// from each in an array when all threads have completed.  The methods
        /// must take no parameters and have no return value.
        /// </summary>
        /// <param name="m"></param>
        /// <returns></returns>
        public static void For(int fromInclusive, int toExclusive, Method method)
        {
            int count = toExclusive - fromInclusive;
            // Initialize the reset events to keep track of completed threads
            ManualResetEvent[] resetEvents = new ManualResetEvent[count];

            // Launch each method in it's own thread
            for (int i = 0; i < count; i++)
            {
                resetEvents[i] = new ManualResetEvent(false);
                ThreadPool.QueueUserWorkItem(new WaitCallback((object index) =>
                {
                    int methodIndex = (int)index;

                    // Execute the method
                    method(fromInclusive + methodIndex);

                    // Tell the calling thread that we're done
                    resetEvents[methodIndex].Set();
                }), i);
            }

            // Wait for all threads to execute
            WaitHandle.WaitAll(resetEvents);
        }
    }

    /// <summary>
    /// Provides a method for performing a deep copy of an object.
    /// Binary Serialization is used to perform the copy.
    /// </summary>
    public static class ObjectCloner
    {
        /// <summary>
        /// Perform a deep Copy of the object.
        /// </summary>
        /// <typeparam name="T">The type of object being copied.</typeparam>
        /// <param name="source">The object instance to copy.</param>
        /// <returns>The copied object.</returns>
        public static T Clone<T>(T source)
        {
            if (!typeof(T).IsSerializable)
            {
                throw new ArgumentException("The type must be serializable.", "source");
            }

            // Don't serialize a null object, simply return the default for that object
            if (Object.ReferenceEquals(source, null))
            {
                return default(T);
            }

            IFormatter formatter = new BinaryFormatter();
            Stream stream = new MemoryStream();
            using (stream)
            {
                formatter.Serialize(stream, source);
                stream.Seek(0, SeekOrigin.Begin);
                return (T)formatter.Deserialize(stream);
            }
        }

        public static T[] Clone<T>(T source, int numCopies)
        {
            if (!typeof(T).IsSerializable)
            {
                throw new ArgumentException("The type must be serializable.", "source");
            }

            // Don't serialize a null object, simply return the default for that object
            if (Object.ReferenceEquals(source, null))
            {
                return default(T[]);
            }

            IFormatter formatter = new BinaryFormatter();
            Stream stream = new MemoryStream();
            var result = new T[numCopies];
            using (stream)
            {
                formatter.Serialize(stream, source);
                for (int i = 0; i < numCopies; i++)
                {
                    stream.Seek(0, SeekOrigin.Begin);
                    result[i] = (T)formatter.Deserialize(stream);
                }
                return result;
            }
        }
    }


    public static class DerivativeChecker
    {


        public static bool CheckDerivatives(FunctionEval func, double[] x0)
        {
            double eps = 1e-4;
            int K = x0.Length;
            var grad = Vector.Zero(K);
            var dummy = Vector.Zero(K);
            double f0 = func(Vector.FromArray(x0), ref grad); 
            bool allGood = true;
            for (int i = 0; i < K; i++)
            {
                var x = x0.Select(j => j).ToArray();
                double eps2 = eps * Math.Max(Math.Abs(x[i]), 0.1);
                x[i] += eps2;
                double f = func(Vector.FromArray(x), ref dummy);
                double fd = (f - f0) / eps2;
                Console.WriteLine("{2} Analytic gradient: {0} finite difference: {1}", grad[i], fd, i);
            }
            return allGood;
        }
    }


    public static class Utils
    {
        public static PositiveDefiniteMatrix CorrelationFromCovariance(PositiveDefiniteMatrix input)
        {
            var result = new PositiveDefiniteMatrix(input.Rows,input.Cols); 
            for (int i = 0; i < input.Rows; i++)
            {
                for (int j = 0; j < input.Cols; j++)
                {
                    result[i, j] = input[i, j] / (Math.Sqrt(input[i, i]) * Math.Sqrt(input[j, j])); 
                }
            }
            return result; 
        }

        public static Vector[][] MakeGrid(Vector[] X, int gridn)
        {
            string extent; 
            return MakeGrid( X, gridn, out extent); 
        }

        public static Vector[][] MakeGrid(Vector[] X, int gridn, out string extent)
        {
            double[] range0, range1;
            return MakeGrid(X, gridn, out extent, out range0, out range1); 
        }

        public static Vector[][] MakeGrid(Vector[] X, int gridn, out string extent, out double[] range0, out double[] range1)
        {
            var minx0 = X.Select(o => o[0]).Min();
            var maxx0 = X.Select(o => o[0]).Max();
            var rangeOf0 = maxx0 - minx0;
            minx0 -= rangeOf0 / 10.0;
            maxx0 += rangeOf0 / 10.0; 
            var minx1 = X.Select(o => o[1]).Min();
            var maxx1 = X.Select(o => o[1]).Max();
            var rangeOf1 = maxx1 - minx1;
            minx1 -= rangeOf1 / 10.0;
            maxx1 += rangeOf1 / 10.0; 
            extent = "["+minx0+","+maxx0+","+minx1+","+maxx1+"]"; 
            range0 = Utils.Seq(minx0, maxx0, gridn);
            range1 = Utils.Seq(minx1, maxx1, gridn);
            var range1Copy = range1.Select(o => o).ToArray(); 
            return range0.Select(
                i => range1Copy.Select(
                    j => Vector.FromArray(new double[] { i, j })).ToArray()).ToArray();
        }

        public static Gaussian[][] PredictionsOnGrid(Vector[][] xgrid, KernelFunction kf, Vector[] X, Gaussian[] f)
        {
            var Kclone = Utils.GramMatrix(kf, X);
            for (int i = 0; i < X.Length; i++)
            {
                Kclone[i, i] += f[i].GetVariance();
            }
            var spec = Kclone.Inverse();
            return xgrid.Select(o => o.Select(x =>
                 GPPrediction(x, X, f, kf, spec)).ToArray()).ToArray();
        }

        public static Gaussian[] PredictionsOnGrid(Vector[] xgrid, KernelFunction kf, Vector[] X, Gaussian[] f)
        {
            var Kclone = Utils.GramMatrix(kf, X);
            for (int i = 0; i < X.Length; i++)
            {
                Kclone[i, i] += f[i].GetVariance();
            }
            var spec = Kclone.Inverse();
            return xgrid.Select(x =>
                 GPPrediction(x, X, f, kf, spec)).ToArray();
        }

        public static Gaussian GPPrediction(Vector x, Vector[] xData, Gaussian[] y, KernelFunction kf, PositiveDefiniteMatrix spec)
        {
            var KxD = Vector.FromArray(xData.Select(o => kf.EvaluateX1X2(x, o)).ToArray());
            double mean = spec.QuadraticForm(KxD, Vector.FromArray(y.Select(o => o.GetMean()).ToArray()));
            double variance = kf.EvaluateX1X2(x, x) - spec.QuadraticForm(KxD);
            return Gaussian.FromMeanAndVariance(mean, variance);
        }

        public static VectorGaussian CorrelatedPredictionsHelper(Gaussian[][] f, Gaussian[,][] W, Gamma noisePrecisionPost, int Q, int D, int ni)
        {
            var mean = Vector.Zero(D);
            var cov = new PositiveDefiniteMatrix(D, D);
            for (int i = 0; i < D; i++)
            {
                cov[i, i] = noisePrecisionPost.GetMeanInverse();
                for (int k = 0; k < Q; k++)
                {
                    mean[i] += W[i, k][ni].GetMean() * f[k][ni].GetMean();
                    cov[i, i] += W[i, k][ni].GetVariance() * (f[k][ni].GetMean() * f[k][ni].GetMean() + f[k][ni].GetVariance());
                }
                for (int j = 0; j < D; j++)
                {
                    for (int k = 0; k < Q; k++)
                        cov[i, j] += W[i, k][ni].GetMean() * W[j, k][ni].GetMean() * f[k][ni].GetVariance();
                }
            }
            return VectorGaussian.FromMeanAndVariance(mean, cov);
        }

        public static double[] KernelToArray(IKernelFunctionWithParams kf)
        {
            int K = kf.ThetaCount;
            var result = new double[K];
            for (int i = 0; i < K; i++)
            {
                result[i] = kf[i];
            }
            return result;
        }

        public static string[] KernelHyperNames(IKernelFunctionWithParams kf)
        {
            int K = kf.ThetaCount;
            var result = new string[K];
            for (int i = 0; i < K; i++)
            {
                result[i] = kf.IndexToName(i); 
            }
            return result;
        }

        public static VectorGaussian extendByOneDimension(VectorGaussian x, Gaussian marg)
        {
            var mean = x.GetMean();
            var variance = x.GetVariance();
            var newMean = Vector.Zero(x.Dimension + 1);
            var newVar = new PositiveDefiniteMatrix(x.Dimension + 1, x.Dimension + 1);
            for (int i = 0; i < x.Dimension; i++)
            {
                newMean[i] = mean[i];
                for (int j = 0; j < x.Dimension; j++)
                {
                    newVar[i, j] = variance[i, j];
                }
            }
            newMean[x.Dimension] = marg.GetMean();
            newVar[x.Dimension, x.Dimension] = marg.GetVariance();
            return VectorGaussian.FromMeanAndVariance(newMean, newVar);
        }

        public static double[,] NormaliseRows(double[,] x, StreamWriter sw = null, int useFirst = -1, double[] means= null, double[] variances = null)
        {
            var result = new double[x.GetLength(0), x.GetLength(1)];
            if (sw != null)
                sw.WriteLine("mean std");
            if (useFirst == -1)
                useFirst = x.GetLength(1);
            for (int i = 0; i < x.GetLength(0); i++)
            {
                double mean = 0, variance = 0;
                for (int j = 0; j < useFirst; j++)
                {
                    mean += x[i, j];
                    variance += x[i, j] * x[i, j];
                }
                mean = mean / (double)x.GetLength(1);
                variance = (variance / (double)x.GetLength(1) - mean * mean);
                if (sw!=null)
                sw.WriteLine("{0} {1}", mean, Math.Sqrt(variance)); 
                for (int j = 0; j < x.GetLength(1); j++)
                {
                    result[i, j] = (x[i, j] - mean) / Math.Sqrt(variance);
                }
                if (means != null)
                    means[i] = mean;
                if (variances != null)
                    variances[i] = variance; 
            }
            return result;
        }

        public static void CalculateSMSEandMSLL(double[,] test, double[,] train, Gaussian[,] pred, out double SMSE, out double MSLL)
        {
            int D = test.GetLength(0);
            int N = test.GetLength(1);
            SMSE = 0;
            MSLL = 0;
            for (int d = 0; d < D; d++)
            {
                var testGaussian = new GaussianEstimator();
                var trainGaussian = new GaussianEstimator();
                for (int n = 0; n < N; n++)
                {
                    testGaussian.Add(test[d, n]);
                    trainGaussian.Add(train[d, n]);
                }
                var testG = testGaussian.GetDistribution(new Gaussian());
                var trainG = trainGaussian.GetDistribution(new Gaussian());
                double se = 0;
                for (int n = 0; n < N; n++)
                {
                    MSLL += (-pred[d, n].GetLogProb(test[d, n]) - (-trainG.GetLogProb(test[d, n])));
                    double err = pred[d, n].GetMean() - test[d, n];
                    se += err * err;
                }
                SMSE += se / testG.GetVariance();

            }
            MSLL /= (double)(D * N);
            SMSE /= (double)(D * N);
        }

        public static double[] Seq(double first, double last, int n)
        {
            double interval = (last - first) / ((double)n - 1.0);
            return Enumerable.Range(0, n).Select(j => first + interval * j).ToArray();
        }

        public static T[,] JaggedToFlat<T>(T[][] input)
        {
            int rows = input.Length;
            int cols = input[0].Length;
            var result = new T[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = input[i][j];
                }
            }
            return result;
        }

        public static PositiveDefiniteMatrix GramMatrix(IKernelFunction kf, Vector[] xData)
        {
            int nData = xData.Length;

            // Allocate and fill the Kernel matrix.
            PositiveDefiniteMatrix K = new PositiveDefiniteMatrix(nData, nData);

            for (int i = 0; i < nData; i++)
            {
                for (int j = 0; j < nData; j++)
                {
                    // Evaluate the kernel. All hyperparameters, including noise
                    // variance are handled in the kernel.
                    K[i, j] = kf.EvaluateX1X2(xData[i], xData[j]);
                }
            }
            return K;
        }

        public static PositiveDefiniteMatrix GramMatrix(IKernelFunctionWithParams kf, Vector[] xData, ref PositiveDefiniteMatrix[] gradK)
        {
            int nData = xData.Length;

            // Allocate and fill the Kernel matrix.
            PositiveDefiniteMatrix K = new PositiveDefiniteMatrix(nData, nData);

            //gradK = Enumerable.Range(0, kf.ThetaCount).Select(_ => new PositiveDefiniteMatrix(nData, nData)).ToArray();

            for (int i = 0; i < nData; i++)
            {
                for (int j = 0; j < nData; j++)
                {
                    Vector temp = gradK == null ? null : Vector.Zero(kf.ThetaCount);
                    // Evaluate the kernel. All hyperparameters, including noise
                    // variance are handled in the kernel.
                    K[i, j] = kf.EvaluateX1X2(xData[i], xData[j], ref temp);
                    if (gradK != null)
                    for (int t = 0; t < kf.ThetaCount; t++)
                    {
                        gradK[t][i, j] = temp[t];
                    }
                }
            }
            return K;
        }

        public static PositiveDefiniteMatrix GramMatrix(IKernelFunctionWithParams kf, Vector[] xData, int[] hypersToOptimise, ref PositiveDefiniteMatrix[] gradK)
        {
            int nData = xData.Length;

            // Allocate and fill the Kernel matrix.
            PositiveDefiniteMatrix K = new PositiveDefiniteMatrix(nData, nData);

            //gradK = Enumerable.Range(0, kf.ThetaCount).Select(_ => new PositiveDefiniteMatrix(nData, nData)).ToArray();

            for (int i = 0; i < nData; i++)
            {
                for (int j = 0; j < nData; j++)
                {
                    Vector temp = gradK == null ? null : Vector.Zero(kf.ThetaCount);
                    // Evaluate the kernel. All hyperparameters, including noise
                    // variance are handled in the kernel.
                    K[i, j] = kf.EvaluateX1X2(xData[i], xData[j], ref temp);
                    if (gradK != null)
                        for (int t = 0; t < hypersToOptimise.Length; t++)
                        {
                            gradK[t][i, j] = temp[hypersToOptimise[t]];
                        }
                }
            }
            return K;
        }

        public static Vector[] KxD(IKernelFunction kf, Vector[] xData, Vector[] dData)
        {
            int nData = xData.Length;

            var result = new Vector[nData];

            for (int i = 0; i < nData; i++)
            {
                result[i] = Vector.Zero(dData.Length);
                for (int j = 0; j < dData.Length; j++)
                {
                    // Evaluate the kernel. All hyperparameters, including noise
                    // variance are handled in the kernel.
                    result[i][j] = kf.EvaluateX1X2(xData[i], dData[j]);
                }
            }
            return result;
        }

        public static T[][] transpose<T>(T[][] x)
        {
            var result = new T[x[0].Length][];
            for (int i = 0; i < x[0].Length; i++)
            {
                result[i] = new T[x.Length];
                for (int j = 0; j < x.Length; j++)
                {
                    result[i][j] = x[j][i];
                }
            }
            return result;
        }

        public static double[][] ReadMatlabAscii(string filename)
        {
            return File.ReadAllLines(filename)
                .Select(x =>
                {
                    return x.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Select(z => Double.Parse(z)).ToArray();
                })
                .ToArray();
        }

        public static void WriteTable<T>(T[,] input, string fn, Converter<T, string> f)
        {
            using (var sw = new StreamWriter(fn))
            {
                for (int i = 0; i < input.GetLength(0); i++)
                {
                    for (int j = 0; j < input.GetLength(1); j++)
                    {
                        sw.Write(f(input[i, j])+" ");
                    }
                    sw.WriteLine();
                }
            }
        }

        public static double[][] ReadMatlabAscii(string filename, int numSamples)
        {
            return File.ReadAllLines(filename)
                .Select(x =>
                {
                    return x.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Select(z => Double.Parse(z)).ToArray();
                })
                .ToList()
                .GetRange(0, numSamples)
                .ToArray();
        }

        public static Vector[][] ConvertTo1DVectors(double[][] x)
        {
            return x.Select(o => o.Select(p => Vector.Constant(1, p)).ToArray()).ToArray();
        }
    }
}
