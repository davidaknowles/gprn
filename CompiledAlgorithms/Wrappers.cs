using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Distributions.Kernels;


namespace gpnetworkVB
{

    /// <summary>
    /// This class provides convenient wrappers for the compiled algorithms in this project
    /// </summary>
    public class Wrappers
    {

        /// <summary>
        /// Run GPRN with node noise
        /// </summary>
        /// <param name="inputs">Covariates X</param>
        /// <param name="data">Outputs Y</param>
        /// <param name="isMissing">Which elements are missing</param>
        /// <param name="settings">Algorithm settings</param>
        /// <param name="errorMeasureNames"Which error measures to include></param>
        /// <param name="meanFunction">Whether to include a per output mean function</param>
        /// <param name="modelAssessor">Delegate function to assess model performance</param>
        /// <param name="swfn">Filename for logging</param>
        /// <returns>Fitted model</returns>
        public INetworkModel NetworkModelNodeNoiseCA(Vector[] inputs,
            double[,] data,
            bool[,] isMissing,
            Settings settings,
            string[] errorMeasureNames = null,
            bool meanFunction = false,
            Converter<INetworkModel, double[]> modelAssessor = null,
            string swfn = null)
        {

            INetworkModel model ;
            if (meanFunction)
                model = new GPRN_MeanFunction_VMP();
            else
                model = new GPRN_NodeNoise_VMP();

            var nodeOptimiser = new KernelOptimiser(settings);
            var weightOptimiser = new KernelOptimiser(settings);

            nodeOptimiser.xData = inputs;
            weightOptimiser.xData = inputs;

            nodeOptimiser.kernel = ObjectCloner.Clone(settings.node_kernel);
            nodeOptimiser.hypersToOptimise = settings.nodeHypersToOptimise;
            weightOptimiser.kernel = ObjectCloner.Clone(settings.weight_kernel);
            weightOptimiser.hypersToOptimise = settings.weightHypersToOptimise;

            var nodeFunctionsInit = Enumerable.Range(0, settings.Q).Select(i =>
                VectorGaussian.FromMeanAndVariance(
                    VectorGaussian.Sample(Vector.Zero(data.GetLength(1)), PositiveDefiniteMatrix.IdentityScaledBy(data.GetLength(1), 100)),
                    PositiveDefiniteMatrix.IdentityScaledBy(data.GetLength(1), settings.init_precision))).ToArray(); // should put this manually in generated code
            var distArray = Distribution<Vector>.Array(nodeFunctionsInit);

            double inputsRange
                = inputs.Select(i => i[0]).Max() - inputs.Select(i => i[0]).Min();

            Console.WriteLine("Init node kernel {0}\ninit weight kernel {1}", settings.node_kernel, settings.weight_kernel);

            model.SetObservedValue("D", data.GetLength(0));
            model.SetObservedValue("Q", settings.Q);
            model.SetObservedValue("N", data.GetLength(1));
            model.SetObservedValue("observedData", data);
            model.SetObservedValue("nodeFunctionsInitVar", distArray);
            model.SetObservedValue("K_node_inverse", Utils.GramMatrix(nodeOptimiser.kernel, inputs).Inverse());
            model.SetObservedValue("K_weights_inverse", Utils.GramMatrix(weightOptimiser.kernel, inputs).Inverse());
            model.SetObservedValue("noisePrecisionPrior", settings.noisePrecisionPrior);
            if (meanFunction)
            {
                model.SetObservedValue("nodeNoisePrecisionPrior", Enumerable.Range(0, model.Q).Select(o => settings.nodeNoisePrecisionPrior).ToArray());
                model.SetObservedValue("nodeSignalPrecisionsPrior", Enumerable.Range(0, model.Q).Select(o => settings.nodeSignalPrecisionsPrior).ToArray());
                model.SetObservedValue("MeanFunctionKernelFunctions", settings.meanFunction_kernels); 
                model.SetObservedValue("xData", inputs);
                GPFactor.settings = new Settings { solverMethod = settings.solverMethod, ml_tolerance = settings.ml_tolerance }; 
            }
            else
            {
                model.SetObservedValue("nodeNoisePrecisionPrior", settings.nodeNoisePrecisionPrior);
                model.SetObservedValue("nodeSignalPrecisionsPrior", settings.nodeSignalPrecisionsPrior);
            }
            model.SetObservedValue("isMissing", isMissing);

            model.nodeKernelOptimiser = nodeOptimiser;
            model.weightKernelOptimiser = weightOptimiser;

            model.Reset();

            var start = DateTime.Now;

            if (swfn != null)
                using (var sw = new StreamWriter(swfn, true))
                {
                    sw.Write("{0} {1} {2}", "it", "time", "ml");
                    if (errorMeasureNames != null)
                        sw.Write(" " + errorMeasureNames.Aggregate((p, q) => p + " " + q));
                    sw.Write(" " + Utils.KernelHyperNames(nodeOptimiser.kernel).Select(o => "node_" + o).Aggregate((p, q) => p + " " + q));
                    sw.Write(" " + Utils.KernelHyperNames(weightOptimiser.kernel).Select(o => "weight_" + o).Aggregate((p, q) => p + " " + q));
                    sw.Write(" noise");
                    for (int i = 0; i < settings.Q; i++)
                        sw.Write(" signal" + i);
                    for (int i = 0; i < settings.Q; i++)
                        sw.Write(" noise" + i);
                    sw.WriteLine();
                }

            double oldML = double.NegativeInfinity;
            double ml = 0;
            int it = 0;
            for (; it < settings.max_iterations; it++)
            {
                model.Update(1);
                ml = model.Marginal<Bernoulli>("ev").LogOdds;

                var noisePrecisionPost = model.Marginal<Gamma>("noisePrecision");

                var assessment = (modelAssessor != null) ? modelAssessor(model).Select(o => o.ToString()).Aggregate((p, q) => p + " " + q) : "";
                //" node " + nodeOptimiser.kernel + " weight " + weightOptimiser.kernel + "
                Console.WriteLine("It " + it + " ml " + ml + " err  " + assessment); 
                if (Math.Abs(oldML - ml) < settings.ml_tolerance)
                    break;

                oldML = ml;

                if (swfn != null)
                    using (var sw = new StreamWriter(swfn, true))
                    {
                        var nodeSignalPrecisionsPost = model.Marginal<Gamma[]>("nodeSignalPrecisions");
                        var nodeNoisePrecisionsPost = model.Marginal<Gamma[]>("nodeNoisePrecisions");

                        sw.Write("{0} {1} {2}", it, (DateTime.Now - start).TotalMilliseconds, ml);
                        if (modelAssessor != null)
                            sw.Write(" " + assessment);
                        sw.Write(" " + Utils.KernelToArray(nodeOptimiser.kernel).Select(o => o.ToString()).Aggregate((p, q) => p + " " + q));
                        sw.Write(" " + Utils.KernelToArray(weightOptimiser.kernel).Select(o => o.ToString()).Aggregate((p, q) => p + " " + q));
                        sw.Write(" " + noisePrecisionPost.GetMeanInverse());
                        for (int i = 0; i < settings.Q; i++)
                            sw.Write(" " + nodeSignalPrecisionsPost[i].GetMeanInverse());
                        for (int i = 0; i < settings.Q; i++)
                            sw.Write(" " + nodeNoisePrecisionsPost[i].GetMeanInverse());
                        sw.WriteLine();
                    }
            }


            Console.WriteLine("Finished after " + it);

            return model;
        }


        /// <summary>
        /// Run GPRN with node noise and diagonal (rather than isotropic noise)
        /// </summary>
        /// <param name="inputs">Covariates X</param>
        /// <param name="data">Outputs Y</param>
        /// <param name="isMissing">Which elements are missing</param>
        /// <param name="settings">Algorithm settings</param>
        /// <param name="errorMeasureNames"Which error measures to include></param>
        /// <param name="modelAssessor">Delegate function to assess model performance</param>
        /// <param name="swfn">Filename for logging</param>
        /// <returns>Fitted model</returns>
        public GPRN_DiagonalNoise_VMP NetworkModelNodeNoiseCA_Diagonal(Vector[] inputs,
            double[,] data,
            bool[,] isMissing,
            Settings settings,
            string[] errorMeasureNames = null,
            Converter<INetworkModel, double[]> modelAssessor = null,
            string swfn = null)
        {

            var model = new GPRN_DiagonalNoise_VMP();

            var nodeOptimiser = new KernelOptimiser(settings);
            var weightOptimiser = new KernelOptimiser(settings);

            nodeOptimiser.xData = inputs;
            weightOptimiser.xData = inputs;

            nodeOptimiser.kernel = ObjectCloner.Clone(settings.node_kernel);
            nodeOptimiser.hypersToOptimise = settings.nodeHypersToOptimise;
            weightOptimiser.kernel = ObjectCloner.Clone(settings.weight_kernel);
            weightOptimiser.hypersToOptimise = settings.weightHypersToOptimise;

            var nodeFunctionsInit = Enumerable.Range(0, settings.Q).Select(i =>
                VectorGaussian.FromMeanAndVariance(
                    VectorGaussian.Sample(Vector.Zero(data.GetLength(1)), PositiveDefiniteMatrix.IdentityScaledBy(data.GetLength(1), 100)),
                    PositiveDefiniteMatrix.IdentityScaledBy(data.GetLength(1), settings.init_precision))).ToArray(); // should put this manually in generated code
            var distArray = Distribution<Vector>.Array(nodeFunctionsInit);

            double inputsRange
                = inputs.Select(i => i[0]).Max() - inputs.Select(i => i[0]).Min();

            Console.WriteLine("Init node kernel {0}\ninit weight kernel {1}", settings.node_kernel, settings.weight_kernel);

            model.SetObservedValue("D", data.GetLength(0));
            model.SetObservedValue("Q", settings.Q);
            model.SetObservedValue("N", data.GetLength(1));
            model.SetObservedValue("observedData", data);
            model.SetObservedValue("nodeFunctionsInitVar", distArray);
            model.SetObservedValue("K_node_inverse", Utils.GramMatrix(nodeOptimiser.kernel, inputs).Inverse());
            model.SetObservedValue("K_weights_inverse", Utils.GramMatrix(weightOptimiser.kernel, inputs).Inverse());
            model.SetObservedValue("noisePrecisionPrior", settings.noisePrecisionPrior);
            model.SetObservedValue("nodeNoisePrecisionPrior", Enumerable.Range(0, model.Q).Select(o => settings.nodeNoisePrecisionPrior).ToArray());
            model.SetObservedValue("nodeSignalPrecisionsPrior", Enumerable.Range(0, model.Q).Select(o => settings.nodeSignalPrecisionsPrior).ToArray());
            model.SetObservedValue("isMissing", isMissing);

            model.nodeKernelOptimiser = nodeOptimiser;
            model.weightKernelOptimiser = weightOptimiser;

            model.Reset();

            var start = DateTime.Now;

            if (swfn != null)
                using (var sw = new StreamWriter(swfn, true))
                {
                    sw.Write("{0} {1} {2}", "it", "time", "ml");
                    if (errorMeasureNames != null)
                        sw.Write(" " + errorMeasureNames.Aggregate((p, q) => p + " " + q));
                    sw.Write(" " + Utils.KernelHyperNames(nodeOptimiser.kernel).Select(o => "node_" + o).Aggregate((p, q) => p + " " + q));
                    sw.Write(" " + Utils.KernelHyperNames(weightOptimiser.kernel).Select(o => "weight_" + o).Aggregate((p, q) => p + " " + q));
                    for (int i = 0; i < model.D; i++)
                        sw.Write(" noise" + i);
                    for (int i = 0; i < settings.Q; i++)
                        sw.Write(" signal" + i);
                    for (int i = 0; i < settings.Q; i++)
                        sw.Write(" noise" + i);
                    sw.WriteLine();
                }

            double oldML = double.NegativeInfinity;
            double ml = 0;
            int it = 0;
            for (; it < settings.max_iterations; it++)
            {
                model.Update(1);
                ml = model.Marginal<Bernoulli>("ev").LogOdds;

                var noisePrecisionPost = model.Marginal<Gamma[]>("noisePrecisionArray");

                var assessment = (modelAssessor != null) ? modelAssessor(model).Select(o => o.ToString()).Aggregate((p, q) => p + " " + q) : "";

                Console.WriteLine("It " + it + " node " + nodeOptimiser.kernel + " weight " + weightOptimiser.kernel + " ml " + ml + " err  " + assessment);
                if (Math.Abs(oldML - ml) < settings.ml_tolerance)
                    break;

                oldML = ml;

                if (swfn != null)
                    using (var sw = new StreamWriter(swfn, true))
                    {
                        var nodeSignalPrecisionsPost = model.Marginal<Gamma[]>("nodeSignalPrecisions");
                        var nodeNoisePrecisionsPost = model.Marginal<Gamma[]>("nodeNoisePrecisions");

                        sw.Write("{0} {1} {2}", it, (DateTime.Now - start).TotalMilliseconds, ml);
                        if (modelAssessor != null)
                            sw.Write(" " + assessment);
                        sw.Write(" " + Utils.KernelToArray(nodeOptimiser.kernel).Select(o => o.ToString()).Aggregate((p, q) => p + " " + q));
                        sw.Write(" " + Utils.KernelToArray(weightOptimiser.kernel).Select(o => o.ToString()).Aggregate((p, q) => p + " " + q));
                        for (int i = 0; i < model.D; i++)
                            sw.Write(" " + noisePrecisionPost[i].GetMeanInverse());
                        for (int i = 0; i < settings.Q; i++)
                            sw.Write(" " + nodeSignalPrecisionsPost[i].GetMeanInverse());
                        for (int i = 0; i < settings.Q; i++)
                            sw.Write(" " + nodeNoisePrecisionsPost[i].GetMeanInverse());
                        sw.WriteLine();
                    }
            }


            Console.WriteLine("Finished after " + it);

            return model;
        }


        /// <summary>
        /// Run GPRN without node noise
        /// </summary>
        /// <param name="inputs">Covariates X</param>
        /// <param name="data">Outputs Y</param>
        /// <param name="settings">Algorithm settings</param>
        /// <param name="swfn">Filename for logging</param>
        /// <returns>Fitted model</returns>
        public GPRN_VMP NetworkModelCA(Vector[] inputs,
            double[,] data,
            Settings settings,
            string swfn = null)
        {
            bool anyIsMissing = false; // AnyIsMissing(isMissing);

            var model = new GPRN_VMP();

            var nodeOptimiser = new KernelOptimiser(settings);
            var weightOptimiser = new KernelOptimiser(settings);

            nodeOptimiser.xData = inputs;
            weightOptimiser.xData = inputs;

            nodeOptimiser.kernel = ObjectCloner.Clone(settings.node_kernel);
            nodeOptimiser.hypersToOptimise = settings.nodeHypersToOptimise;
            weightOptimiser.kernel = ObjectCloner.Clone(settings.weight_kernel);
            weightOptimiser.hypersToOptimise = settings.weightHypersToOptimise;

            var nodeFunctionsInit = Enumerable.Range(0, settings.Q).Select(i =>
                VectorGaussian.FromMeanAndVariance(
                    VectorGaussian.Sample(Vector.Zero(data.GetLength(1)), PositiveDefiniteMatrix.IdentityScaledBy(data.GetLength(1), 100)),
                    PositiveDefiniteMatrix.IdentityScaledBy(data.GetLength(1), settings.init_precision))).ToArray(); // should put this manually in generated code
            var distArray = Distribution<Vector>.Array(nodeFunctionsInit);

            double inputsRange
                = inputs.Select(i => i[0]).Max() - inputs.Select(i => i[0]).Min();

            Console.WriteLine("Init node kernel {0}\ninit weight kernel {1}", settings.node_kernel, settings.weight_kernel);

            model.SetObservedValue("D", data.GetLength(0));
            model.SetObservedValue("Q", settings.Q);
            model.SetObservedValue("N", data.GetLength(1));
            model.SetObservedValue("observedData", data);
            model.SetObservedValue("nodeFunctionsInitVar", distArray);
            model.SetObservedValue("K_node_inverse", Utils.GramMatrix(nodeOptimiser.kernel, inputs).Inverse());
            model.SetObservedValue("K_weights_inverse", Utils.GramMatrix(weightOptimiser.kernel, inputs).Inverse());
            model.SetObservedValue("noisePrecisionPrior", settings.noisePrecisionPrior);
            //model.SetObservedValue("nodeNoisePrecisionPrior", settings.nodeNoisePrecisionPrior);
            model.SetObservedValue("nodeSignalPrecisionsPrior", settings.nodeSignalPrecisionsPrior);
            //model.SetObservedValue("isMissing", isMissing);

            model.nodeKernelOptimiser = nodeOptimiser;
            model.weightKernelOptimiser = weightOptimiser;

            model.Reset();

            var start = DateTime.Now;

            if (swfn != null)
                using (var sw = new StreamWriter(swfn, true))
                {
                    sw.Write("{0} {1} {2}", "it", "time", "ml");
                    if (anyIsMissing)
                        sw.Write(" {0} {1}", "logProb", "error");
                    sw.Write(" " + Utils.KernelHyperNames(nodeOptimiser.kernel).Select(o => "node_" + o).Aggregate((p, q) => p + " " + q));
                    sw.Write(" " + Utils.KernelHyperNames(weightOptimiser.kernel).Select(o => "weight_" + o).Aggregate((p, q) => p + " " + q));
                    sw.Write(" noise");
                    for (int i = 0; i < settings.Q; i++)
                        sw.Write(" signal" + i);

                    sw.WriteLine();
                }

            double oldML = double.NegativeInfinity;
            double ml = 0;
            int it = 0;
            for (; it < settings.max_iterations; it++)
            {
                model.Update(1);
                ml = model.Marginal<Bernoulli>("ev").LogOdds;
                var noisePrecisionPost = model.Marginal<Gamma>("noisePrecision");
                double logProb = 0, error = 0, MSLL = 0, SMSE = 0;

                Console.WriteLine("It {9} Time: {8:G3} Node ls=exp({0:G3})={1:G3} Weight ls=exp({2:G3})={3:G3} ml={4:G3} error={5:G3} msll={6:G3} smse={7:G3}", nodeOptimiser.kernel[0], Math.Exp(nodeOptimiser.kernel[0]),
                    weightOptimiser.kernel[0], Math.Exp(weightOptimiser.kernel[0]), ml, error, MSLL, SMSE, (DateTime.Now - start).TotalMilliseconds, it);
                if (Math.Abs(oldML - ml) < settings.ml_tolerance)
                    break;

                oldML = ml;

                if (swfn != null)
                    using (var sw = new StreamWriter(swfn, true))
                    {
                        var nodeSignalPrecisionsPost = model.Marginal<Gamma[]>("nodeSignalPrecisions");

                        sw.Write("{0} {1} {2}", it, (DateTime.Now - start).TotalMilliseconds, ml);
                        if (anyIsMissing)
                            sw.Write(" {0} {1}", logProb, error);
                        sw.Write(" " + Utils.KernelToArray(nodeOptimiser.kernel).Select(o => o.ToString()).Aggregate((p, q) => p + " " + q));
                        sw.Write(" " + Utils.KernelToArray(weightOptimiser.kernel).Select(o => o.ToString()).Aggregate((p, q) => p + " " + q));
                        sw.Write(" " + noisePrecisionPost.GetMeanInverse());
                        for (int i = 0; i < settings.Q; i++)
                            sw.Write(" " + nodeSignalPrecisionsPost[i].GetMeanInverse());
                        sw.WriteLine();
                    }
            }


            Console.WriteLine("Finished after " + it);

            return model;
        }



        /// <summary>
        /// Run our VB implementation of the Semi Parametric Latent Factor Model of 
        /// Teh, Y., Seeger, M., and Jordan, M. (AISTATS 2005).
        /// </summary>
        public SPLFM_VMP RunSPLFM_VMP(Vector[] inputs,
           double[,] data,
           bool[,] isMissing,
           Settings settings,
           string[] errorMeasureNames = null,
           Converter<IPredictionSPLFMModel, double[]> modelAssessor = null,
           string swfn = null)
        {

            var model = new SPLFM_VMP();

            var nodeOptimiser = new KernelOptimiser(settings);

            nodeOptimiser.xData = inputs;

            nodeOptimiser.kernel = ObjectCloner.Clone(settings.node_kernel);
            nodeOptimiser.hypersToOptimise = settings.nodeHypersToOptimise;

            var nodeFunctionsInit = Enumerable.Range(0, settings.Q).Select(i =>
                VectorGaussian.FromMeanAndVariance(
                    VectorGaussian.Sample(Vector.Zero(data.GetLength(1)), PositiveDefiniteMatrix.IdentityScaledBy(data.GetLength(1), 100)),
                    PositiveDefiniteMatrix.IdentityScaledBy(data.GetLength(1), settings.init_precision))).ToArray(); // should put this manually in generated code
            var distArray = Distribution<Vector>.Array(nodeFunctionsInit);

            double inputsRange
                = inputs.Select(i => i[0]).Max() - inputs.Select(i => i[0]).Min();

            Console.WriteLine("Init node kernel {0}", settings.node_kernel);

            model.SetObservedValue("D", data.GetLength(0));
            model.SetObservedValue("Q", settings.Q);
            model.SetObservedValue("N", data.GetLength(1));
            model.SetObservedValue("observedData", data);
            model.SetObservedValue("nodeFunctionsInitVar", distArray);
            model.SetObservedValue("K_node_inverse", Utils.GramMatrix(nodeOptimiser.kernel, inputs).Inverse());
            model.SetObservedValue("noisePrecisionPrior", settings.noisePrecisionPrior);
            //model.SetObservedValue("nodeNoisePrecisionPrior", settings.nodeNoisePrecisionPrior);
            model.SetObservedValue("nodeSignalPrecisionsPrior", Enumerable.Range(0, settings.Q).Select(o => settings.nodeSignalPrecisionsPrior).ToArray());
            model.SetObservedValue("isMissing", isMissing);

            model.nodeKernelOptimiser = nodeOptimiser;

            model.Reset();

            var start = DateTime.Now;

            if (swfn != null)
                using (var sw = new StreamWriter(swfn, true))
                {
                    sw.Write("{0} {1} {2}", "it", "time", "ml");
                    if (errorMeasureNames != null)
                        sw.Write(" " + errorMeasureNames.Aggregate((p, q) => p + " " + q));
                    sw.Write(" " + Utils.KernelHyperNames(nodeOptimiser.kernel).Select(o => "node_" + o).Aggregate((p, q) => p + " " + q));

                    sw.Write(" noise");
                    for (int i = 0; i < settings.Q; i++)
                        sw.Write(" signal" + i);
                    sw.WriteLine();
                }

            double oldML = double.NegativeInfinity;
            double ml = 0;
            int it = 0;
            for (; it < settings.max_iterations; it++)
            {
                model.Update(1);
                ml = model.Marginal<Bernoulli>("ev").LogOdds;

                var noisePrecisionPost = model.Marginal<Gamma>("noisePrecision");

                var assessment = (modelAssessor != null) ? modelAssessor(model).Select(o => o.ToString()).Aggregate((p, q) => p + " " + q) : "";

                Console.WriteLine("It " + it + " node " + nodeOptimiser.kernel + " ml " + ml + " err  " + assessment);
                if (Math.Abs(oldML - ml) < settings.ml_tolerance)
                    break;

                oldML = ml;

                if (swfn != null)
                    using (var sw = new StreamWriter(swfn, true))
                    {
                        var nodeSignalPrecisionsPost = model.Marginal<Gamma[]>("nodeSignalPrecisions");

                        sw.Write("{0} {1} {2}", it, (DateTime.Now - start).TotalMilliseconds, ml);
                        if (modelAssessor != null)
                            sw.Write(" " + assessment);
                        sw.Write(" " + Utils.KernelToArray(nodeOptimiser.kernel).Select(o => o.ToString()).Aggregate((p, q) => p + " " + q));
                        sw.Write(" " + noisePrecisionPost.GetMeanInverse());
                        for (int i = 0; i < settings.Q; i++)
                            sw.Write(" " + nodeSignalPrecisionsPost[i].GetMeanInverse());
                        sw.WriteLine();
                    }
            }


            Console.WriteLine("Finished after " + it);

            return model;
        }



    }
}
