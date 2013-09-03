using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Distributions.Kernels; 

namespace gpnetworkVB
{
    [Serializable]
    [System.Xml.Serialization.XmlInclude(typeof(SummationKernel))]
    public class Settings
    {
        public bool isotropicNoise = true; 
        public double scaling; 
        public KernelFunction node_kernel;
        public string node_kernel_str
        {
            get
            {
                return node_kernel.ToString();
            }
            set
            {

            }
        }
        public KernelFunction weight_kernel;
        public string weight_kernel_str
        {
            get
            {
                return weight_kernel == null ? "" : weight_kernel.ToString();
            }
            set
            {

            }
        }

        public KernelFunction[] meanFunction_kernels;

        public int max_iterations = 200; // maximum outer iterations of VB to do
        public double ml_tolerance = .1; // tolerance of change in marginal likelihood

        public int iterationsBeforeOptimiseHypers = 0; 

        /// <summary>
        /// which solver to use: MySolver is cheaper per VB iteration, MSF is more accurate
        /// </summary>
        public enum SolverMethod { MySolver, GradientDescent } ;
        public SolverMethod solverMethod = SolverMethod.GradientDescent;

        /// <summary>
        /// Precision of the symmetry breaking initialisation on the node functions 
        /// </summary>
        public double init_precision = 100.0;
        public int[] weightHypersToOptimise; // which weight covariance function hypers to optimise
        public int[] nodeHypersToOptimise; // which node covariance function hypers to optimise

        /// <summary>
        /// Prior on the precision of the noise on the observations 
        /// </summary>
        public Gamma noisePrecisionPrior = Gamma.FromShapeAndRate(1, .1);

        /// <summary>
        /// Prior on the precision of the noise on the nodes (if any)
        /// </summary>
        public Gamma nodeNoisePrecisionPrior = Gamma.FromShapeAndRate(.1, .01);

        /// <summary>
        /// Prior on the precision of the node signal (i.e. inverse sqrt amplitude)
        /// This is sparse to do something like ARD as a proxy for model selection
        /// over the number of latent functions. 
        /// </summary>
        public Gamma nodeSignalPrecisionsPrior = Gamma.FromShapeAndRate(.1, .1);

        /// <summary>
        /// Whether the node functions should have noise added
        /// </summary>
        public bool nodeNoise = false;

        /// <summary>
        /// How many latent functions to use 
        /// </summary>
        public int Q = 1;
    }
}
