using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Distributions.Kernels;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Utils;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer;

namespace gpnetworkVB
{
    public static class TestGPFactor
    {


        public static void Test()
        {
            var inputs = Enumerable.Range(0, 50).Select(i => Vector.Constant(1, i)).ToArray();
            var data = inputs.Select(j => Math.Cos(2 * j[0] / 10.0)).ToArray();
            var n = new Range(data.Length);
            //var kf = new SummationKernel(new ARD(new double[]{ 0 }, 0))+new WhiteNoise();
            var kf = new SummationKernel(new SquaredExponential()) + new WhiteNoise();
            var y = Variable<Vector>.Factor<double, Vector[], int[], KernelFunction>(MyFactors.GP, 1.0/*Variable.GammaFromShapeAndRate(1,1)*/, inputs, new int[] { 0, 1 },
                kf);
            GPFactor.settings = new Settings
            {
                solverMethod = Settings.SolverMethod.GradientDescent,
            };
            y.AddAttribute(new MarginalPrototype(new VectorGaussian(n.SizeAsInt)));
            var y2 = Variable.ArrayFromVector(y, n);
            y2.ObservedValue = data;
            var ypredictive = Variable.ArrayFromVector(y, n);
            var ie = new InferenceEngine(new VariationalMessagePassing());
            var post = ie.Infer<Gaussian[]>(ypredictive);

            var mplWrapper = new MatplotlibWrapper();
            mplWrapper.AddArray("x", inputs.Select(j => j[0]).ToArray());
            mplWrapper.AddArray("y", data);
            var f = post.Select(i => i.GetMean()).ToArray();
            var e = post.Select(i => Math.Sqrt(i.GetVariance())).ToArray();
            mplWrapper.AddArray("f", f);
            mplWrapper.AddArray("e", e);

            mplWrapper.Plot(new string[] {
                "fill_between(x,f-e,f+e,color=\"gray\")",
            "scatter(x,y)"});
        }

        public static void TestWithNoise()
        {
            var inputs = Enumerable.Range(0, 50).Select(i => Vector.Constant(1, i)).ToArray();
            var data = inputs.Select(j => Math.Cos(2 * j[0] / 10.0)).ToArray();
            TestWithNoise(inputs, data);
        }

        public static void TestWithNoise(Vector[] inputs, double[] data)
        {

            var n = new Range(data.Length);
            //var kf = new SummationKernel(new ARD(new double[]{ 0 }, 0))+new WhiteNoise();
            var kf = new SummationKernel(new SquaredExponential()) + new WhiteNoise(-3);
            var y = Variable<Vector>.Factor<double, Vector[], int[], KernelFunction>(MyFactors.GP, 1.0/*Variable.GammaFromShapeAndRate(1,1)*/, inputs, new int[] { 0, 1 },
                kf);
            GPFactor.settings = new Settings
            {
                solverMethod = Settings.SolverMethod.GradientDescent,
            };

            var kf_noise = new SummationKernel(new SquaredExponential()) + new WhiteNoise(-3);
            var noiseFunction = Variable<Vector>.Factor<double, Vector[], int[], KernelFunction>(MyFactors.GP, 1.0/*Variable.GammaFromShapeAndRate(1,1)*/, inputs, new int[] { 0, 1 },
                kf_noise);
            GPFactor.settings = new Settings
            {
                solverMethod = Settings.SolverMethod.GradientDescent,
            };
            noiseFunction.AddAttribute(new MarginalPrototype(new VectorGaussian(n.SizeAsInt)));
            var noiseFunctionValues = Variable.ArrayFromVector(noiseFunction, n);
            var noisePrecisionValues = Variable.Array<double>(n);
            //noisePrecisionValues[n] = Variable.Exp(noiseFunctionValues[n] + 2.0); 
            noisePrecisionValues[n] = Variable.Exp(noiseFunctionValues[n] + Variable.GaussianFromMeanAndPrecision(0, 1));

            y.AddAttribute(new MarginalPrototype(new VectorGaussian(n.SizeAsInt)));
            var y2noiseless = Variable.ArrayFromVector(y, n);
            var y2 = Variable.Array<double>(n);
            y2[n] = Variable.GaussianFromMeanAndPrecision(y2noiseless[n], noisePrecisionValues[n]);
            y2.ObservedValue = data;
            var ypredictiveNoiseless = Variable.ArrayFromVector(y, n);
            var ypredictive = Variable.Array<double>(n);
            ypredictive[n] = Variable.GaussianFromMeanAndPrecision(ypredictiveNoiseless[n], noisePrecisionValues[n]);
            var ie = new InferenceEngine(new VariationalMessagePassing());
            var post = ie.Infer<Gaussian[]>(ypredictive);

            var mplWrapper = new MatplotlibWrapper();
            mplWrapper.AddArray("x", inputs.Select(j => j[0]).ToArray());
            mplWrapper.AddArray("y", data);
            var f = post.Select(i => i.GetMean()).ToArray();
            var e = post.Select(i => 2.0 * Math.Sqrt(i.GetVariance())).ToArray();
            mplWrapper.AddArray("f", f);
            mplWrapper.AddArray("e", e);

            mplWrapper.Plot(new string[] {
                "fill_between(x,f-e,f+e,color=\"gray\")",
                "plot(x,f,'k')",
            "scatter(x,y)"});
        }
    }
}
