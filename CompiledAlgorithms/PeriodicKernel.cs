
using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Utils;
using MicrosoftResearch.Infer.Distributions.Kernels;

namespace gpnetworkVB
{
    /// <summary>
    /// k(x_1,x_2) = exp(-2* [sin(PI |x_1 - x_2|) / periodicity] ^ 2)
    /// </summary>
    [Serializable]
    public class PeriodicKernel : KernelFunction
    {
        private static int version = 1;   // version for read/write
        public double periodicity
        {
            get {
                return base[0]; 
            }
            set {
                base[0] = value; 
            }
        }

        [Construction("periodicity")]
        public PeriodicKernel(double periodicity)
            : base(new string[] { "periodicity" })
        {
            this.periodicity = periodicity;
        }

        public override string ToString()
        {
            return "Periodic(" + periodicity +")";
        }

        #region IKernelFunction Members

        /// <summary>
        /// Evaluates the kernel for a pair of vectors
        /// </summary>
        /// <param name="x1">First vector</param>
        /// <param name="x2">Second vector</param>
        /// <param name="x1Deriv">Derivative of the kernel value with respect to x1 input vector</param>
        /// <param name="logThetaDeriv">Derivative of the kernel value with respect to the log hyper-parameters</param>
        /// <returns></returns>
        public override double EvaluateX1X2(Vector x1, Vector x2, ref Vector x1Deriv, ref Vector logThetaDeriv)
        {
            if (object.ReferenceEquals(x1, x2))
            {
                return EvaluateX(x1, ref x1Deriv, ref logThetaDeriv);
            }
            else
            {
                Vector dvec = Vector.Zero(x1.Count);
                dvec.SetToDifference(x1, x2);
                if (((object)logThetaDeriv) != null)
                {
                    logThetaDeriv[0] = 0; 
                }
                if (((object)x1Deriv) != null)
                {
                    throw new ApplicationException("not implemented yet"); 
                }
                double temp = Math.Sin(Math.PI * Math.Sqrt( dvec.Inner(dvec) ) / periodicity); 
                return Math.Exp(-2.0 * temp * temp);
            }
        }

        /// <summary>
        /// Evaluates the kernel for a single vector (which is used for both slots)
        /// </summary>
        /// <param name="x">Vector</param>
        /// <param name="xDeriv">Derivative of the kernel value with respect to x</param>
        /// <param name="logThetaDeriv">Derivative of the kernel value with respect to the log hyper-parameters</param>
        /// <returns></returns>
        public override double EvaluateX(Vector x, ref Vector xDeriv, ref Vector logThetaDeriv)
        {
            if (((object)logThetaDeriv) != null)
            {
                logThetaDeriv[0] = 0; 
            }
            if (((object)xDeriv) != null)
            {
                throw new ApplicationException("not implemented yet");
            }
            return 1.0;
        }

        /// <summary>
        /// Sets or gets a log hyper-parameter by index
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public override double this[int index]
        {
            get
            {
                return base[index];
            }
            set
            {
                if (index == 0)
                {
                    periodicity = value;
                }
            }
        }

        /// <summary>
        /// The static version for the derived class
        /// </summary>
        public override int TypeVersion
        {
            get
            {
                return version;
            }
        }
        #endregion
    }
}
