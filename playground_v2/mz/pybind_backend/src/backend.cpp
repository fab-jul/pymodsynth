
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <algorithm>


constexpr float pi = 3.14159265358979323846;


/******************************************************************************/
/* Filters                                                                    */
/******************************************************************************/


void butterworth_lowpass_or_highpass(
  // Input signal, shape (num_samples,)
  py::array_t<float> signal_arr,
  // Cutoff freqs, shape (num_samples,)
  py::array_t<float> cutoffs_arr,
  // Output array, will write here.
  py::array_t<float> result_arr,
  // Window coefficients, shape (n,)
  py::array_t<float> w0_arr,
  py::array_t<float> w1_arr,
  py::array_t<float> w2_arr,
  float sampling_rate,
  // This toggles low pass or high pass behavior.
  bool low_pass
) {
  /** Butterworth Lowpass OR Highpass filter.

  The cutoff frequency may be different for each sample.
  */

  py::buffer_info signal_buf = signal_arr.request();
  py::buffer_info cutoffs_buf = cutoffs_arr.request();
  py::buffer_info result_buf = result_arr.request();

  int num_samples = signal_buf.shape[0];

  // Get pointers of signals.
  float *signal =  (float *) signal_buf.ptr,
        *result =  (float *) result_buf.ptr,
        *cutoffs = (float *) cutoffs_buf.ptr;

  // Get pointers for w0, w1, w2.
  py::buffer_info w0_buf = w0_arr.request(),
                  w1_buf = w1_arr.request(),
                  w2_buf = w2_arr.request();
  float *w0 = (float *) w0_buf.ptr,
        *w1 = (float *) w1_buf.ptr,
        *w2 = (float *) w2_buf.ptr;

  // This is ==order/2.
  int n = w0_buf.shape[0];
  int actual_n = n*2;

  // Current output.
  float x = 0;
  float w1sign = low_pass ? 1 : -1;

  float x0 = 0,   // Holds x[i]
        x1 = 0,   // Holds x[i-1]
        x2 = 0;   // Holds x[i-2]

  float xk0 = 0,   // Holds x_k[i]
        xk1 = 0,   // Holds x_k[i-1]
        xk2 = 0;   // Holds x_k[i-w]

  float *y0 = w0,
        *y1 = w1,
        *y2 = w2;

  for (size_t sample_i = 0; sample_i < num_samples; sample_i++) {
    x = signal[sample_i];
    const float cutoff_i = cutoffs[sample_i];
    
    /* Discretize butterworth via second-order sections.

    Based on [1].

    We limit ourselfs to n even. We can factor the transfer function H_n,lp(s) of a
    n-th order butterworth lowpass into a product of second order sections (see [1]):

      H_n,lp(s) = \prod_{k=0}^{n/2-1} H_2k(s),
    
    As shown in [1], H_2k(s) becomes the following in the z-domain: 

                1  + 2 z^-1  + z^-2            Y(z)
      H_2k(z) = ---------------------------- = ----
                b0_k + b1_k z^-1 + b2_k z^-2   X(z)

    where b0_k = gamma^2 - alpha_k gamma + 1,
          b1_k = 2 - 2\gamma^2
          b2_k = gamma^2 + alpha_k gamma + 1,

          gamma, alpha_k as in [1].

    Converting H_2k(s) into discrete space, we land at:

      b0_k y_k[i] + b1_k y_k[i-1] + b2_k y_k[i-2] 
                  = x_k[i] + 2 x_k[i-1] + x_k[i-2]  (1.)
      b0_k y_k[i] = - b1_k y_k[i-1] - b2_k y_k[i-2] + x_k[i] + 2 x_k[i-1] + x_k[i-2] 
           y_k[i] = 1/b0_k * (- b1_k y_k[i-1] - b2_k y_k[i-2] + x_k[i] + 2 x_k[i-1] + x_k[i-2])

    Here, `i` indexes the time dimension (i.e., it goes from t to t+num_samples).
    Note the sub `k` at both y and x, this is due to using SOS. Our input is given as
    x_in[n] and our real output we want is y_out[n]. We can calulate y_out by iterating Eq 1. 
    At each step, we set x_k[i] = y_{k-1}[i], hwere x_0[i] = x_in[i].
    The final y_out becomes y_n.

    For the _high pass_, we have that

     H_n,hp(s) = s^n * H_n,lp(s)

    and thus

     H_2k,hp(s) = H_2k(s) * s^2

    TODO: Derive code! 
  
    References
    ----------

    [1]: https://tttapa.github.io/Pages/Mathematics/Systems-and-Control-Theory/Digital-filters/Discretization/Discretization-of-a-fourth-order-Butterworth-filter.html

    */

    x0 = signal[sample_i];
    const float gammaF = 1/(std::tan(pi*cutoff_i/sampling_rate));
    // We set xk[t] = x[t] for i = {t, t-1, t-2}.
    xk0 = x0;
    xk1 = x1;
    xk2 = x2;
    for (size_t k=0; k <= (actual_n/2-1); k++){
      const float alpha_k = 2 * std::cos(2*pi * (2.*k + actual_n + 1)/(4.*actual_n));
      const float b0k = gammaF * gammaF - alpha_k * gammaF + 1.;
      const float b1k = 2. - 2. * gammaF*gammaF;
      const float b2k = gammaF * gammaF + alpha_k * gammaF + 1.;
      y0[k] = (-b1k*y1[k] - b2k*y2[k] + xk0 + 2. * xk1 + xk2) / b0k;

      xk0 = y0[k];
      xk1 = y1[k];
      xk2 = y2[k];

      // Shift ys
      y2[k] = y1[k];
      y1[k] = y0[k];
    }

    x2 = x1;
    x1 = x0;

    result[sample_i] = y0[actual_n/2-1];
  }
}

/******************************************************************************/
/* Exporting to Python                                                        */
/******************************************************************************/


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Backend";
    m.def("butterworth_lowpass_or_highpass", &butterworth_lowpass_or_highpass,
          "Butterworth Low Pass OR High Pass");
}
