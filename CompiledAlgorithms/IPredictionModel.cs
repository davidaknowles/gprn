using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MicrosoftResearch.Infer;

namespace gpnetworkVB
{

    /// <summary>
    /// Common interface for the GPRN model variants
    /// </summary>
    public interface INetworkModel : IGeneratedAlgorithm
    {
        int Q { get; }
        int D { get; }
        int N { get; }

        bool HasMeanFunctions();  

        KernelOptimiser nodeKernelOptimiser { get; set; }
        KernelOptimiser weightKernelOptimiser { get; set; }
    }

    /// <summary>
    /// Interface for the SPLF models (there is only actually one in 
    /// </summary>
    public interface IPredictionSPLFMModel : IGeneratedAlgorithm
    {
        int Q { get; }
        int D { get; }
        int N { get; }
    }
}
