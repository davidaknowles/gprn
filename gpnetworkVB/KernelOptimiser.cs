using System;
using MicrosoftResearch.Infer;
using System.Linq;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Distributions.Kernels;
using gpnetworkVB;
using System.Threading.Tasks; 
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Collections;
using MicrosoftResearch.Infer.Factors;

namespace gpnetworkVB
{
    /// <summary>
    /// Class to optimise kernel hyperparameters of a Gaussian process
    /// </summary>
    [Serializable]
    public class KernelOptimiser
    {
        [NonSerialized]
        Settings settings;

        public KernelOptimiser(Settings s)
        {
            settings = s;
        }
        [NonSerialized]
        public Vector[] xData;

        public KernelFunction kernel;

        /// <summary>
        /// A list of the indices of which hypers we would like to optimise
        /// </summary>
        [NonSerialized]
        public int[] hypersToOptimise;

        /// <summary>
        /// Count of how many function evaluations we've used
        /// </summary>
        [NonSerialized]
        private int callsCounter = 0;

        /// <summary>
        /// Explicitly calculate the inverse covariance (precision matrix), and optionally the derivatives
        /// (of the covariance matrix) with respect to each of the hyperparameters being optimised
        /// </summary>
        /// <param name="hypers"></param>
        /// <param name="gradK"></param>
        /// <returns></returns>
        public PositiveDefiniteMatrix GramPrecision(double[] hypers, ref PositiveDefiniteMatrix[] gradK)
        {
            for (int i = 0; i < hypersToOptimise.Length; i++)
            {
                kernel[hypersToOptimise[i]] = hypers[i];
            }
            return Utils.GramMatrix(kernel, xData, hypersToOptimise, ref gradK).Inverse();
        }

        /// <summary>
        /// Function delegate for the marginal likelihood
        /// </summary>
        /// <param name="prec">Precision (inverse covariance)</param>
        /// <param name="gradK">Derivatives of the covariance matrix K wrt each kernel parameter</param>
        /// <param name="gradientVector">Vector in which the gradient wrt each kernel parameter should be returned. 
        /// Null if this is not required</param>
        /// <returns></returns>
        public delegate double MarginalLikelihoodFunction(PositiveDefiniteMatrix prec, PositiveDefiniteMatrix[] gradK, Vector gradientVector);

        /// <summary>
        /// Try to optimise the kernel parameters under the specified marginal likelihood function. Note that only 
        /// one step is taken since this will be called within a VB algorithm so it is inefficient to run to
        /// convergence
        /// </summary>
        /// <param name="marginalLikelihoodFunction"></param>
        /// <param name="result">The resulting precision matrix (to be cached for efficiency)</param>
        public void Optimise(MarginalLikelihoodFunction marginalLikelihoodFunction,
            ref PositiveDefiniteMatrix result)
        {
            callsCounter++;
            if (callsCounter <= settings.iterationsBeforeOptimiseHypers || hypersToOptimise.Length == 0)
                return; 
            var startingPoint = hypersToOptimise.Select(i => kernel[i]).ToArray();
            
            switch (settings.solverMethod)
            {
                case Settings.SolverMethod.MySolver:
                    if (hypersToOptimise.Length > 1)
                        throw new ApplicationException("MySolver cannot currently optimise in more than 1D, please use LBFGS or GradientDescent"); 
                    Converter<double, double> f3 = ll =>
                    {
                        kernel[hypersToOptimise[0]] = ll;
                        var prec = Utils.GramMatrix(kernel, xData).Inverse();
                        return marginalLikelihoodFunction(prec, null, null);
                    };
                    double maxLoc = My1DOptimiser(f3, kernel[hypersToOptimise[0]]);
                    kernel[hypersToOptimise[0]] = maxLoc;
                    //Console.WriteLine("ll {0}, likelihood went from {1} to {2}", kernel[0], f3(startingPoint[0]), f3(kernel[0]));
                    result=Utils.GramMatrix(kernel, xData).Inverse();
                    return;

                case Settings.SolverMethod.GradientDescent:
                    FunctionEval fgd = delegate(Vector hypers, ref Vector gradientVector)
                    {
                        PositiveDefiniteMatrix[] gradK = (gradientVector == null) ?
                            null :
                            System.Linq.Enumerable.Range(0, hypers.Count).Select(_ => new PositiveDefiniteMatrix(xData.Length, xData.Length)).ToArray();
                        var prec = GramPrecision(hypers.ToArray(), ref gradK);
                        return marginalLikelihoodFunction(prec, gradK, gradientVector);
                    };
                    Vector gradientVect = Vector.Zero(hypersToOptimise.Length), dummy= null; 
                    var startPointerVector = Vector.FromArray(startingPoint);
                    // take a small step in the steepest ascent direction
                    double f0 = fgd(startPointerVector, ref gradientVect);
                    double eps = .1; 
                    var absg = Math.Sqrt(gradientVect.Sum(o => o * o));
                    var xnew = Vector.FromArray(startingPoint) + gradientVect * (eps / absg);
                    // find the function value here
                    double fnew = fgd(xnew, ref  dummy);
                    // use this to approximate the curvature under a quadratic assumption
                    double m = (fnew - f0) / (eps * eps) - absg / eps;
                    double fopt; 
                    if (m < -1e-2) // good, m negative implies we have a convex function!
                    {
                        // go to the minimum of our quadratic approximation
                        double step = 1.0;
                        var xopt = Vector.FromArray(startingPoint) - gradientVect * (step / (2.0 * m));
                        fopt = fgd(xopt, ref  dummy);
                        // keep back tracking until the function improves
                        while (fopt < f0)
                        {
                            step *= .5;
                            xopt = Vector.FromArray(startingPoint) - gradientVect * (step / (2.0 * m));
                            fopt = fgd(xopt, ref  dummy);
                            if (step < 1.0e-10)
                                throw new ApplicationException("step size was tiny");
                        }
                    }
                    else
                    {
                        if (fnew <= f0)
                        {
                            Console.WriteLine("Seem to be at the minimum! Gradient=" + absg);
                            return;
                        }
                        double step = 1.0;
                        double stepVal = fnew;
                        step *= 2.0;
                        double nextStepVal = fgd(startPointerVector + gradientVect * (step * eps / absg), ref  dummy);
                        // keep going until marginal likelihood decreases
                        while (nextStepVal > stepVal)
                        {
                            step *= 2.0;
                            stepVal = nextStepVal;
                            nextStepVal = fgd(startPointerVector + gradientVect * (step * eps / absg), ref  dummy);
                            if (step > 1e6)
                                throw new ApplicationException("Kernel hyperparameter optimisation diverged"); 
                        }
                        step /= 2.0;
                        fopt=fgd(startPointerVector + gradientVect * (step * eps / absg), ref  dummy);
                        if (fopt < f0)
                            throw new ApplicationException("Kernel hyperparameter optimisation diverged"); 
                    }
                    
                    result = Utils.GramMatrix(kernel, xData).Inverse();
                    Console.WriteLine("ll {0}, likelihood went from {1} to {2}", kernel[0], f0, fopt);
                    if (double.IsNaN(fopt) || fopt < f0)
                        throw new ApplicationException(); 
                    return; 
            }
            throw new ApplicationException(); 
        }



        /// <summary>
        /// Routine to take a gradient step attempting to maximise a 1D function
        /// </summary>
        /// <param name="f">Function to be maximised</param>
        /// <param name="x0">Starting point</param>
        /// <returns>End point (note only one step is taken!)</returns>
        private static double My1DOptimiser(Converter<double, double> f, double x0)
        {
            double eps = 0.1;
            var xPlusMinus = new double[] { x0 - eps, x0 + eps };

            // fit quadratic (take current x=0)
            var L = xPlusMinus.Select(x => f(x)).ToArray();
            double b = (L[1] - L[0]) / (2.0 * eps);
            double f0 = f(x0); 
            double m = (b * eps + L[0] - f0) / (eps * eps);
            if (m < 0) // good! 
            {
                double maxLoc = x0 - b / (2.0 * m); // optimum of quadratic
                // if maxLoc is within eps of x0 just take the maximum of x0 and maxLoc
                if (Math.Abs(maxLoc - x0) < eps)
                {
                    if (f(maxLoc) > f0)
                        return maxLoc;
                    else
                        return x0;
                }
                // LM steps: reduce step size exponentially until improvement is made. This is
                // basically to prevent limit cycles
                double factor = 1.0;
                while (f(maxLoc) < f0)
                {
                    factor /= 2.0;
                    maxLoc = x0 - factor * b / (2.0 * m);
                    if (factor < 1e-10)
                        throw new ApplicationException();
                }
                return maxLoc;
            }
            else // non convex. 
            {
                Console.WriteLine("Warning: nonconvex in log length scale, m=" + m);
                double step = L[1] > L[0] ? eps : -eps;
                double stepLoc = x0 + step;
                double stepVal = Math.Max(L[0], L[1]);
                step *= 2.0;
                double nextStepVal = f(x0 + step);
                // keep going until marginal likelihood decreases
                while (nextStepVal > stepVal)
                {
                    step *= 2.0;
                    stepVal = nextStepVal;
                    nextStepVal = f(x0 + step);
                }
                return x0 + step / 2.0;
            }
        }


        public static double mlfHelperNode(PositiveDefiniteMatrix prec, PositiveDefiniteMatrix[] gradK, Vector gradientVector, Vector[] functions_SampleMean,
            PositiveDefiniteMatrix[] functions_SampleVariance,
            DistributionStructArray<Gamma, double> nodeSignalPrecisions)
        {
            var logDetPrec = VectorGaussianScaledPrecisionOp.PrecisionMeanLogDet(prec);
            double res = 0;
            if (gradientVector != null)
                for (int i = 0; i < gradientVector.Count; i++)
                {
                    gradientVector[i] = 0;
                }

            for (int q = 0; q < functions_SampleMean.Length; q++)
            {
                res += VectorGaussianScaledPrecisionOp.AverageLogFactor(functions_SampleMean[q], nodeSignalPrecisions[q],
                    VectorGaussianScaledPrecisionOp.ESamplePrecisionSample(functions_SampleMean[q], functions_SampleVariance[q], prec),
                    logDetPrec);
                if (gradientVector != null)
                    for (int i = 0; i < gradientVector.Count; i++)
                    {
                        gradientVector[i] += VectorGaussianScaledPrecisionOp.GradAverageLogFactor(
                            functions_SampleMean[q],
                            functions_SampleVariance[q],
                            nodeSignalPrecisions[q],
                            gradK[i],
                            prec);
                    }
            }
            return res;
        }

        public static double mlfHelperWeight(PositiveDefiniteMatrix prec, PositiveDefiniteMatrix[] gradK, Vector gradientVector, Vector[][] functions_SampleMean,
           PositiveDefiniteMatrix[][] functions_SampleVariance)
        {
            double res = 0;
            var logDetPrec = VectorGaussianScaledPrecisionOp.PrecisionMeanLogDet(prec);
            if (gradientVector != null)
                for (int i = 0; i < gradientVector.Count; i++)
                {
                    gradientVector[i] = 0;
                }
            int D= functions_SampleMean[0].Length; 
            var resArray = new double[D]; 
            Vector[] gradVArray = gradientVector != null ? System.Linq.Enumerable.Range(0, D).Select(_ => DenseVector.Zero(gradientVector.Count)).ToArray() : null; 
            for (int q = 0; q < functions_SampleMean.Length; q++)
            {    //for (int d = 0; d < functions_SampleMean[0].Length; d++)
                
                Parallel.For(0, D, d =>
                {
                    resArray[d] = VectorGaussianScaledPrecisionOp.AverageLogFactor(functions_SampleMean[q][d], 1,
                        VectorGaussianScaledPrecisionOp.ESamplePrecisionSample(functions_SampleMean[q][d], functions_SampleVariance[q][d], prec),
                        logDetPrec);
                    if (gradientVector != null)
                        for (int i = 0; i < gradientVector.Count; i++)
                        {
                            gradVArray[d][i] = VectorGaussianScaledPrecisionOp.GradAverageLogFactor(
                                functions_SampleMean[q][d],
                                functions_SampleVariance[q][d],
                                Gamma.PointMass(1),
                                gradK[i],
                                prec);
                        }
                });
            }
            res = resArray.Sum(); 
            if (gradientVector != null)
                gradientVector.SetTo(gradVArray.Aggregate((p,q)=>p+q)); 
            return res;
        }
    }
}
