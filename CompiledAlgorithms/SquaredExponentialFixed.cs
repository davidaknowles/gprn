﻿
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
    /// Squared Exponential kernel function: k(x,y) = exp(-0.5*(x-y)^2/exp(2*logLength))
    /// </summary>
    [Serializable]
    public class SquaredExponentialFixed : KernelFunction
    {
        private static int version = 1;   // version for read/write
        private double lenMult;
        private double signalVar;

        /// <summary>
        /// Constructs the kernel k(x,y) = exp(2*logSignalSD - 0.5*(x-y)^2/exp(2*logLength))
        /// </summary>
        /// <param name="logLengthScale">Log length</param>
        /// <param name="logSignalSD">Log signal variance</param>
        [Construction("LogLengthScale", "LogSignalSD")]
        public SquaredExponentialFixed(double logLengthScale, double logSignalSD)
            : base(new string[] { "Length", "SignalSD" })
        {
            this.LogLengthScale = logLengthScale;
            this.LogSignalSD = logSignalSD;
        }

        /// <summary>
        /// Constructs the kernel k(x,y) = exp(- 0.5*(x-y)^2/exp(2*logLength))
        /// </summary>
        /// <param name="logLengthScale"></param>
        public SquaredExponentialFixed(double logLengthScale)
            : this(logLengthScale, 0.0)
        {
        }

        /// <summary>
        /// Constructs the kernel k(x,y) = exp(- 0.5*(x-y)^2)
        /// </summary>
        public SquaredExponentialFixed()
            : this(0.0, 0.0)
        {
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return "SquaredExponential(" + LogLengthScale + "," + LogSignalSD + ")";
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
                double d = lenMult * dvec.Inner(dvec);
                double de = Math.Exp(d);
                double result = signalVar * de;
                if (((object)logThetaDeriv) != null)
                {
                    logThetaDeriv[0] = -2.0 * result * d;
                    logThetaDeriv[1] = 2.0 * signalVar * de;
                }
                if (((object)x1Deriv) != null)
                {
                    x1Deriv.SetToProduct(dvec, result * 2.0 * lenMult);
                }
                return result;
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
            double result = signalVar;
            if (((object)logThetaDeriv) != null)
            {
                logThetaDeriv[0] = 0.0;
                logThetaDeriv[1] = 2.0 * signalVar; 
            }
            if (((object)xDeriv) != null)
            {
                xDeriv.SetAllElementsTo(0.0);
            }
            return result;
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
                    LogLengthScale = value;
                }
                else if (index == 1)
                {
                    LogSignalSD = value;
                }
            }
        }

        /// <summary>
        /// Sets/gets log of the length scale
        /// </summary>
        public double LogLengthScale
        {
            get
            {
                return 0.5 * Math.Log(-0.5 / lenMult);
            }
            set
            {
                double len = Math.Exp(value);
                lenMult = -0.5 / (len * len);
                base[0] = value;
            }
        }

        /// <summary>
        /// Gets/sets log of the signal variance
        /// </summary>
        public double LogSignalSD
        {
            get
            {
                return 0.5 * Math.Log(signalVar);
            }
            set
            {
                signalVar = Math.Exp(2 * value);
                base[1] = value;
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
