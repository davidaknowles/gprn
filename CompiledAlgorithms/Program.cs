using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using System.Threading.Tasks;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Distributions.Kernels;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Utils;
namespace gpnetworkVB
{
    class Program
    {
        static void Main(string[] args)
        {
            // args should be xfile yfile xstar ystar numFactors numIterations initLengthScales
            Rand.Restart(System.DateTime.Now.Millisecond);
            //RunCommandLine(args);
            Jura(normalise: true, logTransform: true, meanFunctions: false, numRepeats: 1); 
            //EquityPredictions(); 


            Console.Read();
        }


        static void RunCommandLine(string[] args)
        {

            var usageString = "Usage is: gprn x y missing q iterations flls wlls \n" +
                "where x, y and missing are csv or tab delimited text files, x is N x P, y is N x D, and missing is N x P. \n" +
                "x are vector of P inputs at the N datapoints, y is the outputs, missing is a binary matrix of which \n" +
                "elements of y should be considered as missing. q is the number of latent factors to use. \n" +
                "iterations is the maximum number of iterations to run. flls specifies the initial log length scale\n" +
                "for the functions, and wlls for the weights. Note that only the spherical squared exponential \n" +
                "kernel is available through the command line interface.";
            if (args.Length < 5)
            {
                Console.WriteLine("Not enough inputs specified.\n " + usageString);
                return; 
            }
            var xfile = args[0];
            var yfile = args[1];
            var missingfile = args[2];
            var q = int.Parse(args[3]);
            var iterations = int.Parse(args[4]);
            var flls = double.Parse(args[5]);
            var wlls = double.Parse(args[6]);


            var X = File.ReadAllLines(xfile)
                .Select<string, Vector>(i => Vector.FromArray(i.Split(new char[] { ' ', ',', '\t' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(j => double.Parse(j)).ToArray())).ToArray();

            var y_jagged = File.ReadAllLines(yfile)
                .Select<string, double[]>(i => i.Split(new char[] { ' ', ',', '\t' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(j => double.Parse(j)).ToArray()).ToArray();

            var y = Utils.JaggedToFlat(Utils.transpose(y_jagged));

            var missing_jagged = File.ReadAllLines(missingfile).Select<string, bool[]>(i => i.Split(new char[] { ' ', ',', '\t' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(j => double.Parse(j) == 1.0 ? true : false).ToArray()).ToArray();

            var missing = Utils.JaggedToFlat(Utils.transpose(missing_jagged));

            var settings = new Settings
            {
                solverMethod = Settings.SolverMethod.GradientDescent,
                node_kernel = new SummationKernel(new SquaredExponential(flls)) + new WhiteNoise(-3),
                weight_kernel = new SummationKernel(new SquaredExponential(wlls)) + new WhiteNoise(-3),
                meanFunction_kernels = Enumerable.Range(0, y.GetLength(0)).Select(o => new SummationKernel(new SquaredExponential(0)) + new WhiteNoise(-3)).ToArray(),
                nodeHypersToOptimise = new int[] { 0 }, // just optimise the lengthscales
                weightHypersToOptimise = new int[] { 0 },
                nodeNoise = true,
                isotropicNoise = true,
                Q = q,
                iterationsBeforeOptimiseHypers = 10,
                max_iterations = iterations
            };

            double[] means = null;
            double[] variances = null;
            bool normalise = true;
            if (normalise)
            {
                means = new double[y.GetLength(0)];
                variances = new double[y.GetLength(0)];
                y = Utils.NormaliseRows(y, means: means, variances: variances);
            }

            var resultsDir = @"results_" + DateTime.Now.Year + DateTime.Now.Month + DateTime.Now.Day + DateTime.Now.Hour + DateTime.Now.Minute;
            Directory.CreateDirectory(resultsDir);

            


            Converter<INetworkModel, double[]> modelAssessor = m =>
            {
                return Utilities.AssessOnMissing(m, y, missing, false, means, variances);
            };
            var errorMeasureNames = new string[] { "logProb", "error" };
            if (settings.isotropicNoise)
            {
                var model = (new Wrappers()).NetworkModelNodeNoiseCA(X, y, missing, settings, meanFunction: false, modelAssessor: modelAssessor, errorMeasureNames: errorMeasureNames, swfn: resultsDir + @"/results.txt");
            }
            else
            {
                var model = (new Wrappers()).NetworkModelNodeNoiseCA_Diagonal(X, y, missing, settings, modelAssessor: modelAssessor, errorMeasureNames: errorMeasureNames, swfn: resultsDir + @"/results.txt");
            }

        }


        /// <summary>
        /// Wrapper function for multivariate volatility experiments. This can calculate predictive covariances either 
        /// just for the training data (times 1-training), or it can do one step ahead predictions for "steps" steps. 
        /// </summary>
        /// <param name="s">Algorithm settings</param>
        /// <param name="resultsDir">Where to save results</param>
        /// <param name="dataFileName">Text file containing training data</param>
        /// <param name="scaling">Rescaling of the data</param>
        /// <param name="normalisation">Whether to normalise the data</param>
        /// <param name="training">Consider [1,training] as training data</param>
        /// <param name="steps">How many look ahead predictive steps to do</param>
        /// <param name="historicalOnly">Should we only calculate covariances on the training data?</param>
        static void MultivariateVolatilityPredictions(Settings s,
            string resultsDir,
            string dataFileName,
            double scaling = 1.0,
            bool normalisation = true,
            int training = 200,
            int steps = 200,
            bool historicalOnly = false)
        {

            // Load data
            var X = Enumerable.Range(0, training).Select(i => Vector.Constant(1, i)).ToArray();
            var dataJagged = File.ReadAllLines(dataFileName).Take(training)
                .Select<string, double[]>(i => i.Split(new char[] { ' ', ',', '\t' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(j => double.Parse(j) * scaling).ToArray()).ToArray();

            double[,] y = Utils.JaggedToFlat(Utils.transpose(dataJagged));
            if (normalisation)
                using (var sw = new StreamWriter(resultsDir + @"/normalisation.txt"))
                    y = Utils.NormaliseRows(y, sw: sw, useFirst: 200);
            var missing = new bool[y.GetLength(0), y.GetLength(1)];

            // Train model
            var model = (new Wrappers()).NetworkModelNodeNoiseCA(X, y, missing, s, swfn: resultsDir + @"/results_init.txt");

            // calculate predictive covariances on training data
            var histVars = Utilities.HistoricalPredictiveCovariances(model);

            using (var sw = new StreamWriter(resultsDir + @"/historicalPredVars.txt"))
            {
                for (int i = 0; i < X.Length; i++)
                {
                    sw.WriteLine(histVars[i].SourceArray.Select(o => o.ToString()).Aggregate((p, q) => p + " " + q));
                }
            }

            // calculate noise covariances on training data
            var histNoiseVars = Utilities.HistoricalNoiseCovariances(model);
            using (var sw = new StreamWriter(resultsDir + @"/historicalNoiseVars.txt"))
            {
                for (int i = 0; i < X.Length; i++)
                {
                    sw.WriteLine(histNoiseVars[i].SourceArray.Select(o => o.ToString()).Aggregate((p, q) => p + " " + q));
                }
            }

            if (historicalOnly)
                return;

            // Get the variational posterior so we can warm start the training at each look ahead step
            var nodeNoisePrecs = model.Marginal<Gamma[]>("nodeNoisePrecisions").Select(i => i.GetMean()).ToArray();
            var nodeSignalPrecs = model.Marginal<Gamma[]>("nodeSignalPrecisions").Select(i => i.GetMean()).ToArray();
            var obsNoisePrec = model.Marginal<Gamma>("noisePrecision").GetMean();
            var finit = model.Marginal<VectorGaussian[]>("nodeFunctions");
            var winit = model.Marginal<VectorGaussian[,]>("weightFunctions");

            // Do "steps" look ahead steps
            for (int step = 0; step < steps; step++)
            {
                X = Enumerable.Range(0, training + step + 1).Select(i => Vector.Constant(1, i)).ToArray();
                dataJagged = File.ReadAllLines(dataFileName).Take(training + step + 1)
                    .Select<string, double[]>(i => i.Split(new char[] { ' ', ',', '\t' }, StringSplitOptions.RemoveEmptyEntries)
                        .Select(j => double.Parse(j) * scaling).ToArray()).ToArray();
                y = Utils.JaggedToFlat(Utils.transpose(dataJagged));
                if (normalisation)
                    y = Utils.NormaliseRows(y, useFirst: training + step);

                // Rerun training each step but warm start with finit and winit
                VectorGaussian prediction = (new gpnetworkModel()).GPRN_MultivariateVolatility(X, y,
                    nodeSignalPrecs, nodeNoisePrecs, obsNoisePrec, ref finit, ref winit, model.nodeKernelOptimiser.kernel,
                    model.weightKernelOptimiser.kernel);

                using (var sw = new StreamWriter(resultsDir + @"/predictionMean" + step + ".txt"))
                    sw.WriteLine(prediction.GetMean().ToArray().Select(i => i.ToString()).Aggregate((p, q) => p + " " + q));
                using (var sw = new StreamWriter(resultsDir + @"/predictionVar" + step + ".txt"))
                    sw.WriteLine(prediction.GetVariance().SourceArray.Select(i => i.ToString()).Aggregate((p, q) => p + " " + q));
            };
        }

        /// <summary>
        /// Run the model on the equity dataset, calculating covariances on the training data and
        /// performing 200 look ahead covariance predictions 
        /// </summary>
        static void EquityPredictions()
        {
            var baseDir = @"..\..\..\datasets\equity\";
            var resultsDir = @"../../../equity_predictions_new/run" + DateTime.Now.Year + DateTime.Now.Month + DateTime.Now.Day + DateTime.Now.Hour + DateTime.Now.Minute;
            Directory.CreateDirectory(resultsDir);

            var s = new Settings
            {
                solverMethod = Settings.SolverMethod.GradientDescent,
                node_kernel = new SummationKernel(new SquaredExponential(10)) + new WhiteNoise(-3),
                weight_kernel = new SummationKernel(new SquaredExponential(2.683738)) + new WhiteNoise(-3),
                nodeHypersToOptimise = new int[] { },
                weightHypersToOptimise = new int[] { 0 },
                nodeNoise = true,
                Q = 1,
                max_iterations = 200
            };

            MultivariateVolatilityPredictions(s, resultsDir, baseDir + @"data.csv", scaling: Math.Sqrt(15000.0), normalisation: false);
        }



        /// <summary>
        /// Read the Jura data
        /// </summary>
        /// <param name="logTransform">Whether to log  tranform the data</param>
        /// <returns>A tuple of: (covariates, outputs, missngness) </returns>
        public static Tuple<Vector[], double[,], bool[,]> ReadJura(bool logTransform = false)
        {
            var baseDir = @"..\..\..\datasets\jura_csv\";
            var X = File.ReadAllLines(baseDir + @"X.txt")
                .Select<string, Vector>(i => Vector.FromArray(i.Split(new char[] { ' ', ',', '\t' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(j => double.Parse(j)).ToArray())).ToArray();

            var dataJagged = File.ReadAllLines(baseDir + @"Y.txt")
                .Select<string, double[]>(i => i.Split(new char[] { ' ', ',', '\t' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(j => logTransform ? Math.Log(double.Parse(j)) : double.Parse(j)).ToArray()).ToArray();
            var y = Utils.JaggedToFlat(Utils.transpose(dataJagged));
            var missing = new bool[y.GetLength(0), y.GetLength(1)];
            for (int i = 0; i < 100; i++)
                missing[0, 259 + i] = true;
            return new Tuple<Vector[], double[,], bool[,]>(X, y, missing);
        }

        /// <summary>
        /// Run GPRN on the Jura data
        /// </summary>
        /// <param name="normalise">Whether to normalise the data</param>
        /// <param name="logTransform">Whether to log transform the data</param>
        /// <param name="meanFunctions">Whether to include a per output mean function in the model</param>
        /// <param name="numRepeats">How many repeats to run</param>
        static void Jura(bool normalise, bool logTransform, bool meanFunctions, int numRepeats)
        {
            var s = new List<Settings>();

            var data = ReadJura(logTransform: logTransform);
            var X = data.Item1; // covariates
            var y = data.Item2; // outputs

            s.AddRange(Enumerable.Range(0, numRepeats).Select(i => new Settings
            {
                solverMethod = Settings.SolverMethod.GradientDescent,
                node_kernel = new SummationKernel(new ARD(new double[] { 0, 0 }, 0)) + new WhiteNoise(-3),
                weight_kernel = new SummationKernel(new ARD(new double[] { 1, 1 }, 0)) + new WhiteNoise(-3),
                meanFunction_kernels = Enumerable.Range(0, y.GetLength(0)).Select(o => new SummationKernel(new SquaredExponential(0)) + new WhiteNoise(-3)).ToArray(),
                nodeHypersToOptimise = new int[] { 0, 1 },
                weightHypersToOptimise = new int[] { 0, 1 },
                nodeNoise = true,
                isotropicNoise = true,
                Q = 2,
                iterationsBeforeOptimiseHypers = 10
            }));

            double[] means = null;
            double[] variances = null;
            if (normalise)
            {
                means = new double[y.GetLength(0)];
                variances = new double[y.GetLength(0)];
                y = Utils.NormaliseRows(y, means: means, variances: variances);
            }

            var resultsDir = @"../../../jura_results/results_normalise" + normalise + "_logT" + logTransform + "_mf" + meanFunctions + "_" + DateTime.Now.Year + DateTime.Now.Month + DateTime.Now.Day + DateTime.Now.Hour + DateTime.Now.Minute;
            Directory.CreateDirectory(resultsDir);

            var missing = new bool[y.GetLength(0), y.GetLength(1)];
            for (int i = 0; i < 100; i++)
                missing[0, 259 + i] = true;


            Parallel.For(0, s.Count, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, rep =>
            {
                Rand.Restart(rep);
                var xmls = new System.Xml.Serialization.XmlSerializer(typeof(Settings));
                using (var settingssw = new StreamWriter(resultsDir + @"/settings" + rep + ".xml"))
                    xmls.Serialize(settingssw, s[rep]);

                Converter<INetworkModel, double[]> modelAssessor = m =>
                {
                    return Utilities.AssessOnMissing(m, y, missing, logTransform, means, variances);
                };
                var errorMeasureNames = new string[] { "logProb", "error" };
                if (s[rep].isotropicNoise)
                {
                    var model = (new Wrappers()).NetworkModelNodeNoiseCA(X, y, missing, s[rep], meanFunction: meanFunctions, modelAssessor: modelAssessor, errorMeasureNames: errorMeasureNames, swfn: resultsDir + @"/results" + rep + ".txt");
                }
                else
                {
                    if (meanFunctions)
                        throw new ApplicationException("not implemented yet");
                    var model = (new Wrappers()).NetworkModelNodeNoiseCA_Diagonal(X, y, missing, s[rep], modelAssessor: modelAssessor, errorMeasureNames: errorMeasureNames, swfn: resultsDir + @"/results" + rep + ".txt");
                }

            });
        }


    }
}
