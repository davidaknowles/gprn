using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
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
    /// Provides the main GPRN model definition in Infer.NET, and a very simple test routine.
    /// </summary>
    public class gpnetworkModel
    {
        /// <summary>
        /// Simple GPRN test. This will call the Infer.NET compiler so GP hypers will NOT be learnt
        /// </summary>
        /// <param name="nodeNoise">Whether to include node noise</param>
        /// <param name="isotropicNoise">Whether the observation noise is isotropic</param>
        /// <param name="meanFunction">Whether to include a per output mean function</param>
        public void GPRN_Test(bool nodeNoise, bool isotropicNoise, bool meanFunction)
        {
            int N = 2;
            var inputs = Enumerable.Range(0, N).Select(i => Vector.Constant(1, i / 3.0)).ToArray();

            double[,] data = new double[3, N];
            for (int i = 0; i < inputs.Length; i++)
            {
                data[0, i] = Math.Cos(2 * inputs[i][0]);
                data[1, i] = Math.Cos(2 * inputs[i][0]);
                data[2, i] = Math.Sin(2 * inputs[i][0]);
            }

            GPRN_InferNET_model(inputs, data, 2, nodeFunctionNoise: nodeNoise, isotropicNoise: isotropicNoise, meanFunctions: meanFunction, initLoglengthscales: new double[] { 0, 1 });
        }

        /// <summary>
        /// Simple test running Infer.NET on the Semi Parametric Latent Factor Model of 
        /// Teh, Y., Seeger, M., and Jordan, M. (AISTATS 2005).
        /// This will call the Infer.NET compiler so GP hypers will NOT be learnt
        /// </summary>
        public void SPLFM_Test()
        {
            int N = 10;
            var inputs = Enumerable.Range(0, N).Select(i => Vector.Constant(1, i / 3.0)).ToArray();

            double[,] data = new double[3, N];
            for (int i = 0; i < inputs.Length; i++)
            {
                data[0, i] = Math.Cos(2 * inputs[i][0]);
                data[1, i] = Math.Cos(2 * inputs[i][0]);
                data[2, i] = Math.Sin(2 * inputs[i][0]);
            }

            SPLFM(inputs, data, 2);
        }


        /// <summary>
        /// Primary definition of the GPRN model as an Infer.NET model. 
        /// </summary>
        /// <param name="inputs">Covariates X</param>
        /// <param name="data">Outputs Y</param>
        /// <param name="Q">Number of latent functions</param>
        /// <param name="missing">Which elements of Y are missing</param>
        /// <param name="nodeFunctionNoise">Whether to include node noise</param>
        /// <param name="constrainWpositive">Whether to constrain W to be positive [experimental]</param>
        /// <param name="isotropicNoise">Whether to use isotropic observation noise</param>
        /// <param name="meanFunctions">Whether to include a per output mean function</param>
        /// <param name="initLoglengthscales">Initial values for the length scales of the kernels</param>
        /// <param name="sw">An output file for logging</param>
        public void GPRN_InferNET_model(Vector[] inputs,
            double[,] data,
            int Q,
            bool grid = false,
            bool[,] missing = null,
            bool nodeFunctionNoise = false,
            bool constrainWpositive = false,
            bool isotropicNoise = true,
            bool meanFunctions = false,
            double[] initLoglengthscales = null,
            StreamWriter sw = null)
        {
            var toInfer = new List<IVariable>();
            SummationKernel kf_node = new SummationKernel(new SquaredExponential(0)) + new WhiteNoise(-3);
            var K_node = Utils.GramMatrix(kf_node, inputs);

            SummationKernel kf_weights = new SummationKernel(new SquaredExponential(1)) + new WhiteNoise(-3);
            var K_weights = Utils.GramMatrix(kf_weights, inputs);

            var D = Variable.Observed<int>(data.GetLength(0)).Named("D");
            var d = new Range(D).Named("d");
            var Qvar = Variable.Observed<int>(Q).Named("Q");
            var q = new Range(Qvar).Named("q");
            var N = Variable.Observed<int>(data.GetLength(1)).Named("N");
            var n = new Range(N).Named("n");

            if (missing == null)
                missing = new bool[D.ObservedValue, N.ObservedValue]; // check this is all false

            var ev = Variable.Bernoulli(.5).Named("ev");
            var modelBlock = Variable.If(ev);

            var nodeSignalPrecisions = Variable.Array<double>(q).Named("nodeSignalPrecisions");
            // set this to 1 if not learning signal variance
            var nodeSignalPrecisionsPrior = Variable.Observed(Enumerable.Range(0, Q).Select(_ => Gamma.FromShapeAndRate(.1, .1)).ToArray(), q).Named("nodeSignalPrecisionsPrior");
            nodeSignalPrecisions[q] = Variable.Random<double, Gamma>(nodeSignalPrecisionsPrior[q]);

            var nodeFunctions = Variable.Array<Vector>(q).Named("nodeFunctions");
            var K_node_inverse = Variable.Observed(K_node.Inverse()).Named("K_node_inverse");
            nodeFunctions[q] = Variable<Vector>.Factor(MyFactors.VectorGaussianScaled, nodeSignalPrecisions[q], K_node_inverse);
            nodeFunctions.AddAttribute(new MarginalPrototype(new VectorGaussian(N.ObservedValue)));
            var nodeFunctionValues = Variable.Array(Variable.Array<double>(n), q).Named("nodeFunctionValues");
            var nodeFunctionValuesPredictive = Variable.Array(Variable.Array<double>(n), q).Named("nodeFunctionValuesPredictive");

            VariableArray<double> nodeNoisePrecisions = null;
            if (nodeFunctionNoise)
            {
                var nodeFunctionValuesClean = Variable.Array(Variable.Array<double>(n), q).Named("nodeFunctionValuesClean");
                nodeFunctionValuesClean[q] = Variable.ArrayFromVector(nodeFunctions[q], n);
                nodeNoisePrecisions = Variable.Array<double>(q).Named("nodeNoisePrecisions");
                var nodeNoisePrecisionPrior = Variable.Observed(Enumerable.Range(0, Q).Select(_ => Gamma.FromShapeAndRate(.1, .01)).ToArray(), q).Named("nodeNoisePrecisionPrior");
                nodeNoisePrecisions[q] = Variable.Random<double, Gamma>(nodeNoisePrecisionPrior[q]);
                toInfer.Add(nodeNoisePrecisions);
                nodeFunctionValues[q][n] = Variable.GaussianFromMeanAndPrecision(nodeFunctionValuesClean[q][n], nodeNoisePrecisions[q]);

                nodeFunctionValuesPredictive[q][n] = Variable.GaussianFromMeanAndPrecision(nodeFunctionValuesClean[q][n], nodeNoisePrecisions[q]);
            }
            else
            {
                nodeFunctionValues[q] = Variable.ArrayFromVector(nodeFunctions[q], n);
                nodeFunctionValuesPredictive[q] = Variable.ArrayFromVector(nodeFunctions[q], n);
            }

            var weightFunctions = Variable.Array<Vector>(d, q).Named("weightFunctions");
            var K_weights_inverse = Variable.Observed(K_weights.Inverse()).Named("K_weights_inverse");
            weightFunctions[d, q] = Variable<Vector>.Factor(MyFactors.VectorGaussianScaled, Variable.Constant<double>(1), K_weights_inverse).ForEach(d, q);
            weightFunctions.AddAttribute(new MarginalPrototype(new VectorGaussian(N.ObservedValue)));
            var weightFunctionValues = Variable.Array(Variable.Array<double>(n), d, q).Named("weightFunctionValues");
            var weightFunctionValues2 = Variable.Array(Variable.Array<double>(n), d, q).Named("weightFunctionValuesPredictive");
            weightFunctionValues[d, q] = Variable.ArrayFromVector(weightFunctions[d, q], n);
            if (constrainWpositive)
            {
                var weightFunctionValuesCopy = Variable.Array(Variable.Array<double>(n), d, q).Named("weightFunctionValuesCopy");
                weightFunctionValuesCopy[d, q][n] = Variable.GaussianFromMeanAndPrecision(weightFunctionValues[d, q][n], 100);
                Variable.ConstrainPositive(weightFunctionValuesCopy[d, q][n]);
            }
            weightFunctionValues2[d, q] = Variable.ArrayFromVector(weightFunctions[d, q], n);
            var observedData = Variable.Array<double>(d, n).Named("observedData");
            var noisePrecisionPrior = Variable.Observed(Gamma.FromShapeAndRate(1, .1)).Named("noisePrecisionPrior");
            Variable<double> noisePrecision = null;
            VariableArray<double> noisePrecisionArray = null;
            if (isotropicNoise)
            {
                noisePrecision = Variable.Random<double, Gamma>(noisePrecisionPrior).Named("noisePrecision");
                toInfer.Add(noisePrecision);
            }
            else
            {
                noisePrecisionArray = Variable.Array<double>(d).Named("noisePrecision");
                noisePrecisionArray[d] = Variable.Random<double, Gamma>(noisePrecisionPrior).ForEach(d);
                toInfer.Add(noisePrecisionArray);
            }

            var isMissing = Variable.Array<bool>(d, n).Named("isMissing");
            isMissing.ObservedValue = missing;

            var noiseLessY = Variable.Array<double>(d, n).Named("noiseLessY");

            VariableArray<VariableArray<double>, double[][]> meanFunctionValues = null;
            if (meanFunctions)
            {
                GPFactor.settings = new Settings
                    {
                        solverMethod = Settings.SolverMethod.GradientDescent,
                    };

                VariableArray<KernelFunction> kf = Variable.Array<KernelFunction>(d);
                kf.ObservedValue = Enumerable.Range(0, D.ObservedValue).Select(
                    o => new SummationKernel(new SquaredExponential()) + new WhiteNoise(-3)).ToArray();

                var mf = Variable.Array<Vector>(d).Named("meanFunctions");
                mf[d] = Variable<Vector>.Factor<double, Vector[], int[], KernelFunction>(MyFactors.GP, 1.0/*Variable.GammaFromShapeAndRate(1,1)*/, inputs, new int[] { 0 },
                kf[d]);
                mf.AddAttribute(new MarginalPrototype(new VectorGaussian(N.ObservedValue)));
                meanFunctionValues = Variable.Array(Variable.Array<double>(n), d).Named("meanFunctionValues");
                meanFunctionValues[d] = Variable.ArrayFromVector(mf[d], n);
                toInfer.Add(meanFunctionValues);
            }

            using (Variable.ForEach(n))
            using (Variable.ForEach(d))
            {
                var temp = Variable.Array<double>(q).Named("temp");
                temp[q] = weightFunctionValues[d, q][n] * nodeFunctionValues[q][n];
                if (meanFunctions)
                    noiseLessY[d, n] = Variable.Sum(temp) + meanFunctionValues[d][n];
                else
                    noiseLessY[d, n] = Variable.Sum(temp);
                using (Variable.IfNot(isMissing[d, n]))
                    if (isotropicNoise)
                        observedData[d, n] = Variable.GaussianFromMeanAndPrecision(noiseLessY[d, n], noisePrecision);
                    else
                        observedData[d, n] = Variable.GaussianFromMeanAndPrecision(noiseLessY[d, n], noisePrecisionArray[d]);
                using (Variable.If(isMissing[d, n]))
                    observedData[d, n] = Variable.GaussianFromMeanAndPrecision(0, 1);
            }
            observedData.ObservedValue = data;
            var nodeFunctionsInit = Enumerable.Range(0, Q).Select(i =>
                VectorGaussian.FromMeanAndVariance(
                    VectorGaussian.Sample(Vector.Zero(N.ObservedValue), PositiveDefiniteMatrix.IdentityScaledBy(N.ObservedValue, 100)),
                    PositiveDefiniteMatrix.IdentityScaledBy(N.ObservedValue, 100))).ToArray(); // should put this manually in generated code

            var distArray = Distribution<Vector>.Array(nodeFunctionsInit);
            var nodeFunctionsInitVar = Variable.Observed(distArray).Named("nodeFunctionsInitVar");
            nodeFunctions.InitialiseTo(nodeFunctionsInitVar);

            modelBlock.CloseBlock();

            toInfer.AddRange(new List<IVariable>() { ev, noiseLessY, nodeFunctionValues, nodeSignalPrecisions, nodeFunctionValuesPredictive, weightFunctionValues, weightFunctionValues2 });

            var infer = new InferenceEngine(new VariationalMessagePassing());
            infer.ModelName = "MeanFunction";
            var ca = infer.GetCompiledInferenceAlgorithm(toInfer.ToArray());

            var kernel = new SummationKernel(new SquaredExponential(initLoglengthscales[0]));
            kernel += new WhiteNoise(-3);
            ca.SetObservedValue(K_node_inverse.NameInGeneratedCode, Utils.GramMatrix(kernel, inputs).Inverse());

            kernel = new SummationKernel(new SquaredExponential(initLoglengthscales[1]));
            kernel += new WhiteNoise(-3);
            ca.SetObservedValue(K_weights_inverse.NameInGeneratedCode, Utils.GramMatrix(kernel, inputs).Inverse());

            ca.Reset();
            double oldML = double.NegativeInfinity;
            double ml = 0;
            int it = 0;
            for (; it < 100; it++)
            {
                ca.Update(1);
                ml = ca.Marginal<Bernoulli>(ev.NameInGeneratedCode).LogOdds;
                Console.WriteLine(ml);
                if (Math.Abs(oldML - ml) < .1)
                    break;
                oldML = ml;
            }
            Console.WriteLine("Finished after " + it);

        }

        /// <summary>
        /// Infer.NET definition of the Semi Parametric Latent Factor Model of 
        /// Teh, Y., Seeger, M., and Jordan, M. (AISTATS 2005).
        /// </summary>
        /// <param name="inputs">Covariates X</param>
        /// <param name="data">Outputs Y</param>
        /// <param name="Q">Number of latent functions</param>
        /// <param name="missing">Which elements of Y are missing</param>
        /// <param name="nodeFunctionNoise">Whether to include node noise</param>
        public void SPLFM(
            Vector[] inputs, 
            double[,] data, 
            int Q, 
            bool[,] missing = null, 
            bool nodeFunctionNoise = false)
        {
            var toInfer = new List<IVariable>();
            SummationKernel kf_node = new SummationKernel(new SquaredExponential(0));
            var K_node = Utils.GramMatrix(kf_node, inputs);

            var D = Variable.Observed<int>(data.GetLength(0)).Named("D");
            var d = new Range(D).Named("d");
            var Qvar = Variable.Observed<int>(Q).Named("Q");
            var q = new Range(Qvar).Named("q");
            var N = Variable.Observed<int>(data.GetLength(1)).Named("N");
            var n = new Range(N).Named("n");

            if (missing == null)
                missing = new bool[D.ObservedValue, N.ObservedValue]; // check this is all false

            var ev = Variable.Bernoulli(.5).Named("ev");
            var modelBlock = Variable.If(ev);

            var nodeSignalPrecisions = Variable.Array<double>(q).Named("nodeSignalPrecisions");
            // set this to 1 if not learning signal variance
            var nodeSignalPrecisionsPrior = Variable.Observed(Enumerable.Range(0, Q).Select(_ => Gamma.FromShapeAndRate(.1, .1)).ToArray(), q).Named("nodeSignalPrecisionsPrior");
            nodeSignalPrecisions[q] = Variable.Random<double, Gamma>(nodeSignalPrecisionsPrior[q]);

            var nodeFunctions = Variable.Array<Vector>(q).Named("nodeFunctions");
            var K_node_inverse = Variable.Observed(K_node.Inverse()).Named("K_node_inverse");
            nodeFunctions[q] = Variable<Vector>.Factor(MyFactors.VectorGaussianScaled, nodeSignalPrecisions[q], K_node_inverse);
            nodeFunctions.AddAttribute(new MarginalPrototype(new VectorGaussian(N.ObservedValue)));
            var nodeFunctionValues = Variable.Array(Variable.Array<double>(n), q).Named("nodeFunctionValues");
            var nodeFunctionValuesPredictive = Variable.Array(Variable.Array<double>(n), q).Named("nodeFunctionValuesPredictive");

            VariableArray<double> nodeNoisePrecisions = null;
            if (nodeFunctionNoise)
            {
                var nodeFunctionValuesClean = Variable.Array(Variable.Array<double>(n), q).Named("nodeFunctionValuesClean");
                nodeFunctionValuesClean[q] = Variable.ArrayFromVector(nodeFunctions[q], n);
                nodeNoisePrecisions = Variable.Array<double>(q).Named("nodeNoisePrecisions");
                var nodeNoisePrecisionPrior = Variable.Observed(Enumerable.Range(0, Q).Select(_ => Gamma.FromShapeAndRate(.1, .01)).ToArray(), q).Named("nodeNoisePrecisionPrior");
                nodeNoisePrecisions[q] = Variable.Random<double, Gamma>(nodeNoisePrecisionPrior[q]);
                toInfer.Add(nodeNoisePrecisions);
                nodeFunctionValues[q][n] = Variable.GaussianFromMeanAndPrecision(nodeFunctionValuesClean[q][n], nodeNoisePrecisions[q]);

                nodeFunctionValuesPredictive[q][n] = Variable.GaussianFromMeanAndPrecision(nodeFunctionValuesClean[q][n], nodeNoisePrecisions[q]);
            }
            else
            {
                nodeFunctionValues[q] = Variable.ArrayFromVector(nodeFunctions[q], n);
                nodeFunctionValuesPredictive[q] = Variable.ArrayFromVector(nodeFunctions[q], n);
            }

            var weights = Variable.Array<double>(d, q).Named("weights");
            weights[d, q] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(d, q);
            var observedData = Variable.Array<double>(d, n).Named("observedData");
            var noisePrecisionPrior = Variable.Observed(Gamma.FromShapeAndRate(1, .1)).Named("noisePrecisionPrior");
            var noisePrecision = Variable.Random<double, Gamma>(noisePrecisionPrior).Named("noisePrecision");

            var isMissing = Variable.Array<bool>(d, n).Named("isMissing");
            isMissing.ObservedValue = missing;

            var noiseLessY = Variable.Array<double>(d, n).Named("noiseLessY");

            using (Variable.ForEach(n))
            using (Variable.ForEach(d))
            {
                var temp = Variable.Array<double>(q).Named("temp");
                temp[q] = weights[d, q] * nodeFunctionValues[q][n];
                noiseLessY[d, n] = Variable.Sum(temp);
                using (Variable.IfNot(isMissing[d, n]))
                    observedData[d, n] = Variable.GaussianFromMeanAndPrecision(noiseLessY[d, n], noisePrecision);
                using (Variable.If(isMissing[d, n]))
                    observedData[d, n] = Variable.GaussianFromMeanAndPrecision(0, 1);
            }
            observedData.ObservedValue = data;
            var nodeFunctionsInit = Enumerable.Range(0, Q).Select(i =>
                VectorGaussian.FromMeanAndVariance(
                    VectorGaussian.Sample(Vector.Zero(N.ObservedValue), PositiveDefiniteMatrix.IdentityScaledBy(N.ObservedValue, 100)),
                    PositiveDefiniteMatrix.IdentityScaledBy(N.ObservedValue, 100))).ToArray(); // should put this manually in generated code

            var distArray = Distribution<Vector>.Array(nodeFunctionsInit);
            var nodeFunctionsInitVar = Variable.Observed(distArray).Named("nodeFunctionsInitVar");
            nodeFunctions.InitialiseTo(nodeFunctionsInitVar);

            modelBlock.CloseBlock();

            toInfer.AddRange(new List<IVariable>() { ev, noiseLessY, noisePrecision, nodeFunctionValues, nodeSignalPrecisions, nodeFunctionValuesPredictive, weights });

            var ie = new InferenceEngine(new VariationalMessagePassing());
            ie.ModelName = "SPLFM";
            var ca = ie.GetCompiledInferenceAlgorithm(toInfer.ToArray());
            ca.Execute(100);
            var fvals = ca.Marginal<Gaussian[][]>(nodeFunctionValues.NameInGeneratedCode)[0]; // [q][n]
            var x = inputs.Select(i => i[0]).ToArray();
            var mplWrapper = new MatplotlibWrapper();

            mplWrapper.AddArray("x", x);
            mplWrapper.AddArray("y", fvals.Select(i => i.GetMean()).ToArray());
            mplWrapper.AddArray("s", fvals.Select(i => Math.Sqrt(i.GetVariance())).ToArray());

            mplWrapper.Plot(new string[] {
                "fill_between(x,y-s,y+s,color=\"gray\")",
                "ylabel(\"node (fitted)\")"});
        }

        /// <summary>
        /// An implementation of GPRN specialised for one step look ahead multivariate volatility experiments
        /// </summary>
        /// <param name="inputs">Covariates X</param>
        /// <param name="data">Outputs Y</param>
        /// <returns>Predicted covariance for the next time point</returns>
        public VectorGaussian GPRN_MultivariateVolatility(
            Vector[] inputs,
            double[,] data,
            double[] nodeSignalPrecs,
            double[] nodeNoisePrecs,
            double obsNoisePrec,
            ref VectorGaussian[] finit,
            ref VectorGaussian[,] winit,
            KernelFunction nodeKernel,
            KernelFunction weightKernel)
        {
            var missing = new bool[data.GetLength(0), data.GetLength(1)];
            for (int i = 0; i < data.GetLength(0); i++)
            {
                missing[i, data.GetLength(1) - 1] = true; // last data point is missing
            }
            int Q = nodeSignalPrecs.Length;

            var toInfer = new List<IVariable>();
            var K_node = Utils.GramMatrix(nodeKernel, inputs);
            var K_weights = Utils.GramMatrix(weightKernel, inputs);

            var D = Variable.Observed<int>(data.GetLength(0)).Named("D");
            var d = new Range(D).Named("d");
            var Qvar = Variable.Observed<int>(Q).Named("Q");
            var q = new Range(Qvar).Named("q");
            var N = Variable.Observed<int>(data.GetLength(1)).Named("N");
            var n = new Range(N).Named("n");

            var ev = Variable.Bernoulli(.5).Named("ev");
            var modelBlock = Variable.If(ev);

            var nodeSignalPrecisions = Variable.Array<double>(q).Named("nodeSignalPrecisions");
            nodeSignalPrecisions.ObservedValue = nodeSignalPrecs;

            var nodeFunctions = Variable.Array<Vector>(q).Named("nodeFunctions");
            var K_node_inverse = Variable.Observed(K_node.Inverse()).Named("K_node_inverse");
            nodeFunctions[q] = Variable<Vector>.Factor(MyFactors.VectorGaussianScaled, nodeSignalPrecisions[q], K_node_inverse);
            nodeFunctions.AddAttribute(new MarginalPrototype(new VectorGaussian(N.ObservedValue)));
            var nodeFunctionValues = Variable.Array(Variable.Array<double>(n), q).Named("nodeFunctionValues");
            var nodeFunctionValuesPredictive = Variable.Array(Variable.Array<double>(n), q).Named("nodeFunctionValuesPredictive");

            var nodeFunctionValuesClean = Variable.Array(Variable.Array<double>(n), q).Named("nodeFunctionValuesClean");
            nodeFunctionValuesClean[q] = Variable.ArrayFromVector(nodeFunctions[q], n);
            var nodeNoisePrecisions = Variable.Array<double>(q).Named("nodeNoisePrecisions");
            nodeNoisePrecisions.ObservedValue = nodeNoisePrecs;
            nodeFunctionValues[q][n] = Variable.GaussianFromMeanAndPrecision(nodeFunctionValuesClean[q][n], nodeNoisePrecisions[q]);

            nodeFunctionValuesPredictive[q][n] = Variable.GaussianFromMeanAndPrecision(nodeFunctionValuesClean[q][n], nodeNoisePrecisions[q]);

            var weightFunctions = Variable.Array<Vector>(d, q).Named("weightFunctions");
            var K_weights_inverse = Variable.Observed(K_weights.Inverse()).Named("K_weights_inverse");
            weightFunctions[d, q] = Variable<Vector>.Factor(MyFactors.VectorGaussianScaled, Variable.Constant<double>(1), K_weights_inverse).ForEach(d, q);
            weightFunctions.AddAttribute(new MarginalPrototype(new VectorGaussian(N.ObservedValue)));
            var weightFunctionValues = Variable.Array(Variable.Array<double>(n), d, q).Named("weightFunctionValues");
            var weightFunctionValuesPredictive = Variable.Array(Variable.Array<double>(n), d, q).Named("weightFunctionValuesPredictive");
            weightFunctionValues[d, q] = Variable.ArrayFromVector(weightFunctions[d, q], n);

            weightFunctionValuesPredictive[d, q] = Variable.ArrayFromVector(weightFunctions[d, q], n);
            var observedData = Variable.Array<double>(d, n).Named("observedData");
            var noisePrecision = Variable.Observed(obsNoisePrec).Named("noisePrecision");

            var isMissing = Variable.Array<bool>(d, n).Named("isMissing");
            isMissing.ObservedValue = missing;

            var noiseLessY = Variable.Array<double>(d, n).Named("noiseLessY");

            using (Variable.ForEach(n))
            using (Variable.ForEach(d))
            {
                var temp = Variable.Array<double>(q).Named("temp");
                temp[q] = weightFunctionValues[d, q][n] * nodeFunctionValues[q][n];
                noiseLessY[d, n] = Variable.Sum(temp);
                using (Variable.IfNot(isMissing[d, n]))
                    observedData[d, n] = Variable.GaussianFromMeanAndPrecision(noiseLessY[d, n], noisePrecision);
                using (Variable.If(isMissing[d, n]))
                    observedData[d, n] = Variable.GaussianFromMeanAndPrecision(0, 1);
            }
            observedData.ObservedValue = data;
            var nodeFunctionsInit = Enumerable.Range(0, Q).Select(i =>
                VectorGaussian.FromMeanAndVariance(
                    VectorGaussian.Sample(Vector.Zero(N.ObservedValue), PositiveDefiniteMatrix.IdentityScaledBy(N.ObservedValue, 100)),
                    PositiveDefiniteMatrix.IdentityScaledBy(N.ObservedValue, 100))).ToArray(); // should put this manually in generated code

            var distArray = Distribution<Vector>.Array(nodeFunctionsInit);
            var nodeFunctionsInitVar = Variable.Observed(distArray).Named("nodeFunctionsInitVar");
            nodeFunctions.InitialiseTo(nodeFunctionsInitVar);

            var finitNew = finit.Select(i => Utils.extendByOneDimension(i, Gaussian.FromMeanAndVariance(0, 1))).ToArray();
            nodeFunctions.InitialiseTo(Distribution<Vector>.Array(finitNew));

            var winitNew = new VectorGaussian[data.GetLength(0), Q];
            for (int i = 0; i < data.GetLength(0); i++)
            {
                for (int j = 0; j < Q; j++)
                {
                    winitNew[i, j] = Utils.extendByOneDimension(winit[i, j], Gaussian.FromMeanAndVariance(0, 1));
                }
            }

            weightFunctions.InitialiseTo(Distribution<Vector>.Array(winitNew));

            modelBlock.CloseBlock();

            toInfer.AddRange(new List<IVariable>() { ev, noiseLessY, nodeFunctions, weightFunctions, nodeFunctionValuesPredictive, weightFunctionValues, weightFunctionValuesPredictive /* is this redundant? */ });

            var ie = new InferenceEngine(new VariationalMessagePassing());
            var ca = ie.GetCompiledInferenceAlgorithm(toInfer.ToArray());

            ca.SetObservedValue(K_node_inverse.NameInGeneratedCode, Utils.GramMatrix(nodeKernel, inputs).Inverse());
            ca.SetObservedValue(K_weights_inverse.NameInGeneratedCode, Utils.GramMatrix(weightKernel, inputs).Inverse());
            ca.Reset();

            double oldML = double.NegativeInfinity;
            double ml = 0;
            int it = 0;
            for (; it < 30; it++)
            {
                ca.Update(1);
                ml = ca.Marginal<Bernoulli>(ev.NameInGeneratedCode).LogOdds;
                Console.WriteLine(ml);
                if (Math.Abs(oldML - ml) < .1)
                    break;
                oldML = ml;
            }

            var f = ca.Marginal<Gaussian[][]>("nodeFunctionValuesPredictive");
            var W = ca.Marginal<Gaussian[,][]>("weightFunctionValuesPredictive");

            finit = ca.Marginal<VectorGaussian[]>(nodeFunctions.NameInGeneratedCode);
            winit = ca.Marginal<VectorGaussian[,]>(weightFunctions.NameInGeneratedCode);
            return Utils.CorrelatedPredictionsHelper(f, W, Gamma.PointMass(obsNoisePrec), Q, data.GetLength(0), data.GetLength(1) - 1);
        }


    }
}
