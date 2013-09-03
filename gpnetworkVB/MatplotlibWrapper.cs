using System;
using System.IO;
using System.Diagnostics; 
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MicrosoftResearch.Infer.Maths; 

namespace gpnetworkVB
{
    public class MatplotlibWrapper
    {
        List<string> lines = new List<string>();

        public MatplotlibWrapper()
        {
            lines.Add("import numpy");
            lines.Add("from pylab import *");
        }

        public void AddArray(string name, double[] values)
        {
            lines.Add(name + "=numpy.array([" + values.Select(i => i.ToString()).Aggregate((i, j) => i + "," + j) + "],dtype=\"float\")"); 
        }

        public void AddArray2D(string name, double[][] values)
        {
            lines.Add(name + "=numpy.array([" + values.Select(i =>
                "[" + i.Select(k=>k.ToString()).Aggregate((o, j) => o + "," + j) + "]").Aggregate((p,q)=>p+","+q) + "],dtype=\"float\")");
        }

        public void Queue(string[] line)
        {
            lines.AddRange(line);
        }

        private static void OutputHandler(object sendingProcess, DataReceivedEventArgs outLine)
        {
            if (outLine.Data != null)
                Console.WriteLine(outLine.Data.ToString());
        }

        public void Plot(string[] commands = null, string filename= null)
        {
            if (filename == null)
                filename = "temp" + Rand.Int(100) + ".py";
            using (var sw = new StreamWriter(filename))
            {
                foreach (var l in lines)
                    sw.WriteLine(l);
                if (commands != null)
                foreach (var command in commands)
                    sw.WriteLine(command); 
                sw.WriteLine("show()");
            }
            // Start the child process.
            Process p = new Process();
            p.StartInfo.FileName = "pythonw.exe";
            p.StartInfo.Arguments = filename;
            p.StartInfo.UseShellExecute = false;
            p.StartInfo.CreateNoWindow = true;
            p.StartInfo.RedirectStandardOutput = true;
            p.OutputDataReceived += new DataReceivedEventHandler(OutputHandler);
            p.Start();
            p.BeginOutputReadLine();
            p.WaitForExit();
        }
    }
}
