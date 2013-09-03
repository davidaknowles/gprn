using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace gpnetworkVB
{
    class Brent
    {

        public void Test()
        {
            double minLoc = 1.0;
            double minVal = Minimize(x => -x * Math.Exp(-x), -1, 2.0, 1e-6, ref minLoc);
            Console.WriteLine("min={0} at {1}", minVal, minLoc);
        }

        // The return value of Minimize is the minimum of the function f.
        // The location where f takes its minimum is returned in the variable minLoc.
        // Notation and implementation based on Chapter 5 of Richard Brent's book
        // "Algorithms for Minimization Without Derivatives".

        double c = 0.5 * (3.0 - Math.Sqrt(5.0)); // .381
        double SQRT_DBL_EPSILON = Math.Sqrt(double.Epsilon);

        double Minimize
        (
            Converter<double, double> f,		// [in] objective function to minimize
            double leftEnd,     // [in] smaller value of bracketing interval
            double rightEnd,    // [in] larger value of bracketing interval
            double epsilon,     // [in] stopping tolerance
            ref double x      // [out] location of minimum
        )
        {
            double d, e, m, p, q, r, tol, t2, u, v, w, fu, fv, fw, fx;

            double a = leftEnd;
            double b = rightEnd;

            v = w = x = a + c * (b - a);
            d = e = 0.0;
            fv = fw = fx = f(x);
            int counter = 0;

            while (true)
            {
                counter++;
                m = 0.5 * (a + b);
                tol = SQRT_DBL_EPSILON * Math.Abs(x) + epsilon;
                t2 = 2.0 * tol;
                // Check stopping criteria
                if (Math.Abs(x - m) < t2 - 0.5 * (b - a))
                    break;
                p = q = r = 0.0;
                if (Math.Abs(e) > tol)
                {
                    // fit parabola
                    r = (x - w) * (fx - fv);
                    q = (x - v) * (fx - fw);
                    p = (x - v) * q - (x - w) * r;
                    q = 2.0 * (q - r);
                    if (q > 0.0)
                        p = -p;
                    else
                        q = -q;
                    r = e; e = d;
                }
                if (Math.Abs(p) < Math.Abs(0.5 * q * r) && p < q * (a - x) && p < q * (b - x))
                {
                    // A parabolic interpolation step
                    d = p / q;
                    u = x + d;
                    // f must not be evaluated too close to a or b
                    if (u - a < t2 || b - u < t2)
                        d = (x < m) ? tol : -tol;
                }
                else
                {
                    // A golden section step
                    e = (x < m) ? b : a;
                    e -= x;
                    d = c * e;
                }
                // f must not be evaluated too close to x
                if (Math.Abs(d) >= tol)
                    u = x + d;
                else if (d > 0.0)
                    u = x + tol;
                else
                    u = x - tol;
                fu = f(u);
                // Update a, b, v, w, and x
                if (fu <= fx)
                {
                    if (u < x)
                        b = x;
                    else
                        a = x;
                    v = w; fv = fw;
                    w = x; fw = fx;
                    x = u; fx = fu;
                }
                else
                {
                    if (u < x)
                        a = u;
                    else
                        b = u;
                    if (fu <= fw || w == x)
                    {
                        v = w; fv = fw;
                        w = u; fw = fu;
                    }
                    else if (fu <= fv || v == x || v == w)
                    {
                        v = u; fv = fu;
                    }
                }
            }

            return fx;
        }
    }
}
