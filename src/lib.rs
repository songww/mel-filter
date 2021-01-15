use itertools_num::linspace;
use num_traits::{AsPrimitive, Float, NumCast, NumOps, Zero};

pub enum NormalizationFactor {
    /// Leave all the triangles aiming for a peak value of 1.0
    None,
    /// divide the triangular mel weights by the width of the mel band (area normalization).
    One,
    /// Leave all the triangles aiming for a peak value of 1.0
    Inf,
}

pub trait Hz: Float + NumCast {
    /// # Examples
    /// ```
    /// use mel_filter::Hz;
    /// assert_eq!(60.0f64.to_mel(false), 0.8999999999999999);
    /// assert_eq!(0.0f64.to_mel(false), 0.);
    /// assert_eq!(11025.0f64.to_mel(false), 49.91059448015905);
    /// assert_eq!(110.0f64.to_mel(false), 1.65);
    /// assert_eq!(220.0f64.to_mel(false), 3.3);
    /// assert_eq!(440.0f64.to_mel(false), 6.6);
    /// ```
    fn to_mel(&self, htk: bool) -> Self {
        if htk {
            let n = Self::from(2595.0).unwrap();
            return n * (*self / Self::from(700.0).unwrap() + Self::one()).log10();
        }
        // Fill in the linear part
        let f_min = Self::zero();
        let f_sp = Self::from(200.0 / 3.).unwrap();
        let min_log_hz = Self::from(1000.0).unwrap(); // beginning of log region (Hz)
        let min_log_mel = (min_log_hz - f_min) / f_sp; // same (Mels)
        let logstep = Self::from((6.4).ln() / 27.0).unwrap(); // step size for log region
        if self >= &min_log_hz {
            min_log_mel + (*self / min_log_hz).ln() / logstep
        } else {
            (*self - f_min) / f_sp
        }
    }
}

impl Hz for f32 {}
impl Hz for f64 {}

pub trait Mel: Float + NumCast + Zero + Clone {
    ///
    /// # Examples
    ///
    /// ```
    /// use mel_filter::Mel;
    /// assert_eq!(1.0f32.to_hz(false), 66.66666666666667);
    /// assert_eq!(2.0f32.to_hz(false), 133.33333333333334);
    /// assert_eq!(3.0f32.to_hz(false), 200.);
    /// assert_eq!(4.0f32.to_hz(false), 266.66666666666667);
    /// assert_eq!(5.0f32.to_hz(false), 333.3333);
    /// ```
    fn to_hz(&self, htk: bool) -> Self {
        if htk {
            let base: Self = Self::from(10.0).unwrap();
            let seven_hundred: Self = Self::from(700.0).unwrap();
            return seven_hundred * (base.powf(*self / Self::from(2595.0).unwrap()) - Self::one());
        }

        // Fill in the linear scale
        let f_min = Self::zero();
        let f_sp = Self::from(200.0 / 3.0).unwrap();

        // And now the nonlinear scale
        let min_log_hz = Self::from(1000.0).unwrap(); // beginning of log region (Hz)
        let min_log_mel = (min_log_hz - f_min) / f_sp; // same (Mels)
        let logstep = Self::from((6.4).ln() / 27.0).unwrap(); // step size for log region

        // If we have vector data, vectorize
        if self >= &min_log_mel {
            // min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
            min_log_hz * Self::exp(logstep * (*self - min_log_mel))
        } else {
            f_min + f_sp * *self
        }
    }
}

impl Mel for f32 {}
impl Mel for f64 {}

pub trait FilterBank<Mel> {
    fn as_slice(&self) -> &[Mel];
    fn as_mut_slice(&mut self) -> &mut [Mel];
    fn shape(&self) -> &[usize];
    fn zeros(shape: &[usize; 2]) -> Self;
    fn row_mut(&mut self, idx: usize) -> &mut [Mel];
}

#[cfg(feature = "ndarray")]
use ndarray::{prelude::*, Array2};
#[cfg(feature = "ndarray")]
impl<F: Mel + Zero> FilterBank<F> for Array2<F> {
    #[inline]
    fn as_slice(&self) -> &[F] {
        self.as_slice().unwrap()
    }
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [F] {
        Array2::<F>::as_slice_memory_order_mut(self).unwrap()
    }
    #[inline]
    fn shape(&self) -> &[usize] {
        self.shape().as_ref()
    }

    #[inline]
    fn zeros(shape: &[usize; 2]) -> Self {
        Self::zeros(shape.to_owned())
    }

    #[inline]
    fn row_mut(&mut self, idx: usize) -> &mut [F] {
        Array2::<F>::row_mut(self, idx).into_slice().unwrap()
    }
}

pub struct FilterBankMat<Mel> {
    data: Vec<Mel>,
    cols: usize,
    rows: usize,
    shape_: [usize; 2],
}

impl<Mel: Clone + Zero> FilterBank<Mel> for FilterBankMat<Mel> {
    #[inline]
    fn as_slice(&self) -> &[Mel] {
        self.data.as_slice()
    }
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [Mel] {
        self.data.as_mut_slice()
    }
    #[inline]
    fn shape(&self) -> &[usize] {
        &self.shape_
    }

    #[inline]
    fn row_mut(&mut self, idx: usize) -> &mut [Mel] {
        assert!(idx <= self.rows);
        let from = idx * self.cols;
        let to = idx * self.cols + self.cols;
        &mut self.data[from..to]
    }

    #[inline]
    fn zeros(shape: &[usize; 2]) -> Self {
        let rows = shape[0];
        let cols = shape[1];
        Self {
            data: vec![Mel::zero(); rows * cols],
            rows,
            cols,
            shape_: [rows, cols],
        }
    }
}

#[cfg(feature = "torch")]
mod torch_integ {
    use super::*;
    use tch::kind::Kind;
    use tch::IndexOp;
    use tch::Tensor;
    pub trait ToKind {
        fn to_kind() -> Kind;
    }
    impl ToKind for f32 {
        fn to_kind() -> Kind {
            Kind::Float
        }
    }
    impl ToKind for f64 {
        fn to_kind() -> Kind {
            Kind::Double
        }
    }
    impl<M: Mel + ToKind> FilterBank<M> for Tensor {
        fn as_slice(&self) -> &[M] {
            let size = self.size().iter().fold(1, |x, y| x * y);
            unsafe { std::slice::from_raw_parts(self.data_ptr() as *const M, size as usize) }
        }
        fn as_mut_slice(&mut self) -> &mut [M] {
            let size = self.size().iter().fold(1, |x, y| x * y);
            unsafe { std::slice::from_raw_parts_mut(self.data_ptr() as *mut M, size as usize) }
        }
        fn shape(&self) -> &[usize] {
            unimplemented!()
        }
        fn zeros(shape: &[usize; 2]) -> Self {
            let shape = [shape[0] as i64, shape[1] as i64];
            Tensor::zeros(shape.as_ref(), (M::to_kind(), tch::Device::Cpu))
        }
        fn row_mut(&mut self, idx: usize) -> &mut [M] {
            let mut row = self.i(&[idx as i64][..]);
            let size = row.size().iter().fold(1, |x, y| x * y);
            unsafe { std::slice::from_raw_parts_mut(row.data_ptr() as *mut M, size as usize) }
        }
    }
}

#[cfg(feature = "torch")]
pub use torch_integ::*;

/// Implementation of `librosa.hz_to_mel`
///
/// Convert Hz to Mels
///
/// # Parameters
///
/// `frequencies` : number or &[..] , float
///     scalar or slice of frequencies
/// `htk`         : use HTK formula instead of Slaney
///
/// # Returns
/// Vec![..], input frequencies in Mels
///
/// # Examples
///
/// ```
/// use mel_filter::hz_to_mel;
/// assert_eq!(vec![0.8999999999999999], hz_to_mel(&[60.], false));
/// assert_eq!(vec![0.8999999999999999], hz_to_mel(&[60.], false));
/// assert_eq!(vec![0.], hz_to_mel(&[0.], false));
/// assert_eq!(vec![49.91059448015905], hz_to_mel(&[11025.], false));
/// assert_eq!(vec![ 1.65,  3.3 ,  6.6 ], hz_to_mel(vec![110., 220., 440.], false));
/// ```
///
/// # See Also
///
/// [mel_to_hz]
///
pub fn hz_to_mel<T, A>(frequencies: T, htk: bool) -> Vec<A>
where
    T: AsRef<[A]>,
    A: Copy + Float + NumOps + NumCast,
{
    if htk {
        let n: A = NumCast::from(2595.0).unwrap();
        return frequencies
            .as_ref()
            .iter()
            .map(|v| n * (*v / NumCast::from(700.0).unwrap() + NumCast::from(1.0).unwrap()).log10())
            .collect();
    }

    // Fill in the linear part
    let f_min = A::zero();
    let f_sp = NumCast::from(200.0 / 3.).unwrap();

    let mut mels: Vec<A> = frequencies
        .as_ref()
        .iter()
        .map(|v| (*v - f_min) / f_sp)
        .collect();

    // Fill in the log-scale part

    let min_log_hz = NumCast::from(1000.0).unwrap(); // beginning of log region (Hz)
    let min_log_mel = (min_log_hz - f_min) / f_sp; // same (Mels)
    let logstep = NumCast::from((6.4).ln() / 27.0).unwrap(); // step size for log region
                                                             // If we have array data, vectorize
    for (idx, val) in frequencies.as_ref().iter().enumerate() {
        if val >= &min_log_hz {
            mels[idx] = min_log_mel + (*val / min_log_hz).ln() / logstep;
        }
    }
    // min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    mels
    //let log_t = frequencies >= min_log_hz;
    //mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep;
    //
    // If we have scalar data, heck directly
    //min_log_mel + (frequencies / min_log_hz).ln() / logstep;
}

/// Implementation of `librosa.mel_to_hz`
///
/// Convert mel bin numbers to frequencies
///
/// # Parameters
///
/// `mels`          : Vec [shape=(n,)], float mel bins to convert
/// `htk`           : use HTK formula instead of Slaney
///
/// # Returns
///
/// frequencies   : np.ndarray [shape=(n,)]
///     input mels in Hz
///
/// # Examples
///
/// ```
/// use mel_filter::mel_to_hz;
/// assert_eq!(vec![200.], mel_to_hz(&[3.], false));
/// assert_eq!(vec![200.], mel_to_hz(&[3.], false));
///
/// assert_eq!(vec![  66.66666666666667,  133.33333333333334,  200.   ,  266.6666666666667,  333.33333333333337], mel_to_hz(&[1.,2.,3.,4.,5.], false));
/// assert_eq!(vec![  66.66666666666667,  133.33333333333334,  200.   ,  266.6666666666667,  333.33333333333337], mel_to_hz(&[1.,2.,3.,4.,5.], false));
/// ```
///
/// # See Also
///
/// [hz_to_mel]
///
pub fn mel_to_hz<T, A>(mels: T, htk: bool) -> Vec<A>
where
    T: AsRef<[A]>,
    A: Copy + Float + NumOps + NumCast,
{
    if htk {
        let base: A = NumCast::from(10.0).unwrap();
        let seven_hundred: A = NumCast::from(700.0).unwrap();
        return mels
            .as_ref()
            .iter()
            .map(|v| {
                seven_hundred
                    * (base.powf(*v / NumCast::from(2595.0).unwrap()) - NumCast::from(1.0).unwrap())
            })
            .collect();
    }

    let mels = mels.as_ref();

    // Fill in the linear scale
    let f_min = A::zero();
    let f_sp: A = A::from(200.0 / 3.0).unwrap();

    let mut freqs: Vec<A> = mels.iter().map(|v| f_min + f_sp * *v).collect();

    // And now the nonlinear scale
    let min_log_hz = A::from(1000.0).unwrap(); // beginning of log region (Hz)
    let min_log_mel = (min_log_hz - f_min) / f_sp; // same (Mels)
    let logstep = A::from((6.4).ln() / 27.0).unwrap(); // step size for log region

    // If we have vector data, vectorize
    for (idx, val) in mels.iter().enumerate() {
        if val >= &min_log_mel {
            // min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
            freqs[idx] = min_log_hz * A::exp(logstep * (*val - min_log_mel))
        }
    }
    freqs
    /*
    let freqs = f_min + f_sp * mels;
    if mels >= min_log_mel {
        // If we have scalar data, check directly
        freqs = min_log_hz * (logstep * (mels - min_log_mel)).exp();
    }
    freqs
    */
}

/// Implementation of `librosa.mel_frequencies`
///
/// Compute an array of acoustic frequencies tuned to the mel scale.
///
/// The mel scale is a quasi-logarithmic function of acoustic frequency
/// designed such that perceptually similar pitch intervals (e.g. octaves)
/// appear equal in width over the full hearing range.
///
/// Because the definition of the mel scale is conditioned by a finite number
/// of subjective psychoaoustical experiments, several implementations coexist
/// in the audio signal processing literature [#]_. By default, librosa replicates
/// the behavior of the well-established MATLAB Auditory Toolbox of Slaney [#]_.
/// According to this default implementation,  the conversion from Hertz to mel is
/// linear below 1 kHz and logarithmic above 1 kHz. Another available implementation
/// replicates the Hidden Markov Toolkit [#]_ (HTK) according to the following formula::
///
/// >>    mel = 2595.0 * (1.0 + f / 700.0).log10().
///
/// The choice of implementation is determined by the `htk` keyword argument: setting
/// `htk=false` leads to the Auditory toolbox implementation, whereas setting it `htk=true`
/// leads to the HTK implementation.
///
/// - [#] Umesh, S., Cohen, L., & Nelson, D. Fitting the mel scale.
///     In Proc. International Conference on Acoustics, Speech, and Signal Processing
///     (ICASSP), vol. 1, pp. 217-220, 1998.
///
/// - [#] Slaney, M. Auditory Toolbox: A MATLAB Toolbox for Auditory
///     Modeling Work. Technical Report, version 2, Interval Research Corporation, 1998.
///
/// - [#] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., Liu, X.,
///     Moore, G., Odell, J., Ollason, D., Povey, D., Valtchev, V., & Woodland, P.
///     The HTK book, version 3.4. Cambridge University, March 2009.
///
///
/// # See Also
/// [hz_to_mel]
///
/// [mel_to_hz]
///
/// librosa.feature.melspectrogram
///
/// librosa.feature.mfcc
///
/// # Parameters
///
/// `n_mels`    : Number of mel bins.
///
/// `fmin`      : float >= 0, Minimum frequency (Hz).
///
/// `fmax`      : float >= 0, Maximum frequency (Hz).
///
/// `htk`       : bool
///     If True, use HTK formula to convert Hz to mel.
///     Otherwise (False), use Slaney's Auditory Toolbox.
///
/// # Returns
/// `bin_frequencies` : Vec[shape=(n_mels,)]
///     Vector of `n_mels` frequencies in Hz which are uniformly spaced on the Mel
///     axis.
///
/// # Examples
///
/// ```
/// use mel_filter::mel_frequencies;
/// let freqs = mel_frequencies::<f64>(Some(40), None, None, false); // n_mels=40
/// println!("{:?}", freqs);
/// let expected = vec![  0.0             ,   85.31725552163941,  170.63451104327882,
///                     255.95176656491824,  341.26902208655764,  426.586277608197  ,
///                     511.9035331298365 ,  597.2207886514759 ,  682.5380441731153 ,
///                     767.8552996947546 ,  853.172555216394  ,  938.4898107380334 ,
///                    1024.8555458780081 , 1119.1140732107583 , 1222.0417930074345 ,
///                    1334.436032577335  , 1457.1674514162094 , 1591.1867857508237 ,
///                    1737.532213396153  , 1897.3373959769085 , 2071.840260812287  ,
///                    2262.3925904926227 , 2470.4704944333835 , 2697.6858435241707 ,
///                    2945.7987564509885 , 3216.731234416783  , 3512.5820498813423 ,
///                    3835.64300465582   , 4188.416683294875  , 4573.635839312682  ,
///                    4994.284564397706  , 5453.621404613084  , 5955.204602651788  ,
///                    6502.919661685081  , 7101.009444327076  , 7754.107039876304  ,
///                    8467.271654439703  , 9246.027801961029  , 10096.408099746186 ,
///                    11025.0];
/// assert_eq!(freqs, expected);
/// ```
///
pub fn mel_frequencies<T: Hz>(
    n_mels: Option<usize>,
    fmin: Option<T>,
    fmax: Option<T>,
    htk: bool,
) -> Vec<T> {
    let n_mels = n_mels.unwrap_or(128);

    let fmin = fmin.unwrap_or_else(T::zero);

    let fmax = fmax.unwrap_or_else(|| NumCast::from(11025.0).unwrap());

    // 'Center freqs' of mel bands - uniformly spaced between limits
    let min_mel = fmin.to_mel(htk);
    let max_mel = fmax.to_mel(htk);

    let mels: Vec<_> = linspace::<T>(min_mel, max_mel, n_mels).collect();

    mel_to_hz(mels, htk)
}

/// Implementation of `librosa.fft_frequencies`
///
/// Returns Vec of frequencies lenth is `1 + n_fft/2`.
///     Frequencies ``(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)``
///
/// # Parameters
///
/// `sr` : Audio sampling rate
///
/// `n_fft` : FFT window size
///
/// # Examples
///
/// ```
/// use mel_filter::fft_frequencies;
/// assert_eq!(fft_frequencies::<f32>(Some(22050), Some(16)), vec![   0.   ,   1378.125,   2756.25 ,   4134.375,
///                                                                5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.]);
/// assert_eq!(fft_frequencies::<f64>(Some(22050), Some(16)), vec![   0.   ,   1378.125,   2756.25 ,   4134.375,
///                                                                5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.]);
/// // array([     0.   ,   1378.125,   2756.25 ,   4134.375,
/// //          5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ]));
/// ```
///
pub fn fft_frequencies<T: Float + NumOps>(sr: Option<usize>, n_fft: Option<usize>) -> Vec<T> {
    let sr: f64 = sr.unwrap_or(22050).as_();
    let n_fft = n_fft.unwrap_or(2048);

    linspace(T::zero(), NumCast::from(sr / 2.).unwrap(), 1 + n_fft / 2).collect()
}

/// `librosa.filters.mel`.
/// Returns 2D array, of shape `n_mels * (1 + n_fft/2)`, Currently is `Vec<Vec<T>>`.
///
/// # Arguments
///
/// * `sr` - Sampling rate of the incoming signal
/// * `n_fft` - number of FFT components
/// * `n_mels` - number of Mel bands to generate, default 128
/// * `fmin` - lowest frequency (in Hz), default 0.0
/// * `fmax` - highest frequency (in Hz)
///     If `None`, use `fmax = sr / 2.0`
/// * `htk` - use HTK formula instead of Slaney
/// * `norm` if [NormalizationFactor::One], divide the triangular mel weights by the width of the mel band
///     (area normalization).  Otherwise, leave all the triangles aiming for
///     a peak value of 1.0
///
/// # Examples
///
/// ```
/// use mel_filter::{mel, NormalizationFactor, FilterBankMat, FilterBank};
/// let mut melfb = mel::<f64, FilterBankMat<f64>>(22050, 2048, None, None, None, false, NormalizationFactor::One);
/// assert_eq!(melfb.shape(), &[128, 1025]);
/// assert_eq!(&melfb.row_mut(0)[..6], &[0f64, 0.016182853208219942, 0.032365706416439884, 0.028990088037379964, 0.012807234829160026, 0.][..], "begin six element");
/// assert_eq!(&melfb.row_mut(1)[..9], &[0f64, 0., 0., 0.009779235793639925, 0.025962089001859864, 0.035393705451959974, 0.01921085224374004, 0.003027999035520103, 0.][..], "second nine element");
/// // melfb = [[ 0.   ,  0.016, ...,  0.   ,  0.   ],
/// //          [ 0.   ,  0.   , ...,  0.   ,  0.   ],
/// //          ...,
/// //          [ 0.   ,  0.   , ...,  0.   ,  0.   ],
/// //          [ 0.   ,  0.   , ...,  0.   ,  0.   ]]
/// // Clip the maximum frequency to 8KHz
/// let mut melfb: FilterBankMat<f64> = mel(22050, 2048, None, None, Some(8000.), false, NormalizationFactor::One);
/// println!("{:?}", &melfb.row_mut(0)[..10]);
/// assert_eq!(melfb.shape(), &[128, 1025]);
/// assert_eq!(&melfb.row_mut(0)[..6], &[0f64, 0.01969187633619885, 0.0393837526723977, 0.026457473399387796, 0.006765597063188946, 0.][..], "begin six element");
/// assert_eq!(&melfb.row_mut(1)[..9], &[0f64, 0., 0., 0.016309077804604378, 0.036000954140803225, 0.029840271930982275, 0.010148395594783432, 0., 0.][..], "second nine element");
/// // melfb = [[ 0.  ,  0.02, ...,  0.  ,  0.  ],
/// //          [ 0.  ,  0.  , ...,  0.  ,  0.  ],
/// //          ...,
/// //          [ 0.  ,  0.  , ...,  0.  ,  0.  ],
/// //          [ 0.  ,  0.  , ...,  0.  ,  0.  ]])
/// ```
pub fn mel<T: Hz, Out: FilterBank<T>>(
    sr: usize,
    n_fft: usize,
    n_mels: Option<usize>,
    fmin: Option<T>,
    fmax: Option<T>,
    htk: bool,
    norm: NormalizationFactor,
) -> Out {
    let fmax: T = fmax.unwrap_or_else(|| {
        let sr: f64 = sr.as_();
        T::from(sr / 2.).unwrap()
    });

    // Initialize the weights
    let n_mels = n_mels.unwrap_or(128);
    let width = 1 + n_fft / 2;
    let mut weights = Out::zeros(&[n_mels, width]);

    // Center freqs of each FFT bin
    let fftfreqs = fft_frequencies(Some(sr), Some(n_fft));

    // 'Center freqs' of mel bands - uniformly spaced between limits
    let mel_f = mel_frequencies(Some(n_mels + 2), fmin, Some(fmax), htk);

    let fdiff: Vec<_> = mel_f[..mel_f.len() - 1]
        .iter()
        .zip(mel_f[1..].iter())
        .map(|(x, y)| *y - *x)
        .collect();

    // ramps = np.subtract.outer(mel_f, fftfreqs);
    let mut ramps = vec![vec![T::zero(); fftfreqs.len()]; mel_f.len()];

    let _ = mel_f
        .iter()
        .enumerate()
        .map(|(i, m)| {
            fftfreqs
                .iter()
                .enumerate()
                .map(|(j, f)| {
                    ramps[i][j] = *m - *f;
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    for i in 0..n_mels {
        for j in 0..width {
            // lower and upper slopes for all bins
            let lower = -ramps[i][j] / fdiff[i];
            let upper = ramps[i + 2][j] / fdiff[i + 1]; // +2 is safe since we create `mel_f`
                                                        // with `n_mels + 2` size

            // .. then intersect them with each other and zero
            weights.row_mut(i)[j] = T::zero().max(lower.min(upper));
        }
    }

    match norm {
        NormalizationFactor::One => {
            // Slaney-style mel is scaled to be approx constant energy per channel
            // enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
            // weights *= enorm[:, np.newaxis]

            let two: T = NumCast::from(2.).unwrap();
            for (idx, enorm) in mel_f[2..n_mels + 2]
                .iter()
                .zip(mel_f[..n_mels].iter())
                .map(|(x, y)| two / (*x - *y))
                .enumerate()
            {
                let _: Vec<_> = weights
                    .row_mut(idx)
                    .iter_mut()
                    .map(|v| *v = (*v) * enorm)
                    .collect();
            }
            //let enorm = 2.0 / (mel_f[2..n_mels + 2] - mel_f[..n_mels]);
            // weights *= enorm[.., np.newaxis];
        }
        _ => {}
    }

    // Only check weights if f_mel[0] is positive
    //if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)) {
    //    // This means we have an empty channel somewhere
    //    warn!("Empty filters detected in mel frequency basis. Some channels will produce empty responses. Try increasing your sampling rate (and fmax) or reducing n_mels.");
    //}

    weights
}
