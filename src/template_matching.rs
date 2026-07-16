//! Functions for performing template matching.
use crate::definitions::Image;
use image::{GenericImageView, GrayImage, Luma, Primitive};

#[cfg_attr(feature = "katexit", katexit::katexit)]
/// Scoring functions when comparing a template and an image region.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatchTemplateMethod {
    /// Sum of the squares of the difference between image and template pixel intensities. Smaller values indicate a better match.
    ///
    /// Without a mask:
    /// $$
    /// \text{output}(x, y) = \sum_{x', y'} \left( \text{template}(x', y') - \text{image}(x+x', y+y') \right)^2
    /// $$
    ///
    /// With a mask:
    /// $$
    /// \text{output}(x, y) = \sum_{x', y'} \left( (\text{template}(x', y') - \text{image}(x+x', y+y')) \cdot \text{mask}(x', y') \right)^2
    /// $$
    ///
    SumOfSquaredErrors,
    /// Divides the sum computed using `SumOfSquaredErrors` by a normalization term. Smaller values indicate a better match.
    ///
    /// Without a mask:
    /// $$
    /// \text{output}(x, y) = \frac{\sum_{x', y'} \left( \text{template}(x', y') - \text{image}(x+x', y+y') \right)^2}
    ///                     {\sqrt{ \sum_{x', y'} {\text{template}(x', y')}^2 \cdot \sum_{x', y'} {\text{image}(x+x', y+y')}^2 }}
    /// $$
    ///
    /// With a mask:
    /// $$
    /// \text{output}(x, y) = \frac{\sum_{x', y'} \left( (\text{template}(x', y') - \text{image}(x+x', y+y')) \cdot \text{mask}(x', y') \right)^2}
    ///         {\sqrt{ \sum_{x', y'}{(\text{template}(x', y') \cdot \text{mask}(x', y'))}^2 \cdot \sum_{x', y'}{(\text{image}(x+x', y+y') \cdot \text{mask}(x', y'))}^2 }}
    /// $$
    SumOfSquaredErrorsNormalized,
    /// Cross Correlation. Larger values indicate a better match.
    ///
    /// Without a mask:
    /// $$
    /// \text{output}(x, y) = \sum_{x', y'} \left( \text{template}(x', y') \cdot \text{image}(x+x', y+y') \right)
    /// $$
    ///
    /// With a mask:
    /// $$
    /// \text{output}(x, y) = \sum_{x', y'} \left( \text{template}(x', y') \cdot \text{image}(x+x', y+y') \cdot {\text{mask}(x', y')}^2 \right)
    /// $$
    ///
    CrossCorrelation,
    /// Divides the sum computed using `CrossCorrelation` by a normalization term. Larger values indicate a better match.
    ///
    /// Without a mask:
    /// $$
    /// \text{output}(x, y) = \frac{\sum_{x', y'} \left( \text{template}(x', y') \cdot \text{image}(x+x', y+y') \right)}
    ///                     {\sqrt{ \sum_{x', y'} {\text{template}(x', y')}^2 \cdot \sum_{x', y'} {\text{image}(x+x', y+y')}^2 }}
    /// $$
    ///
    /// With a mask:
    /// $$
    /// \text{output}(x, y) = \frac{\sum_{x', y'} \left( \text{template}(x', y') \cdot \text{image}(x+x', y+y') \cdot {\text{mask}(x', y')}^2 \right)}
    ///         {\sqrt{ \sum_{x', y'}{(\text{template}(x', y') \cdot \text{mask}(x', y'))}^2 \cdot \sum_{x', y'}{(\text{image}(x+x', y+y') \cdot \text{mask}(x', y'))}^2 }}
    /// $$
    ///
    CrossCorrelationNormalized,
}

/// Slides a `template` over an `image` and scores the match at each point using
/// the requested `method`.
///
/// The returned image has dimensions `image.width() - template.width() + 1` by
/// `image.height() - template.height() + 1`.
///
/// See [`MatchTemplateMethod`] for details of the matching methods.
///
/// # Panics
///
/// If either dimension of `template` is not strictly less than the corresponding dimension
/// of `image`.
pub fn match_template(
    image: &GrayImage,
    template: &GrayImage,
    method: MatchTemplateMethod,
) -> Image<Luma<f32>> {
    use MatchTemplateMethod as M;

    let input = &ImageTemplate::new(image, template);
    match method {
        M::SumOfSquaredErrors => methods::Sse::match_template(input),
        M::SumOfSquaredErrorsNormalized => methods::SseNormalized::match_template(input),
        M::CrossCorrelation => methods::Ccorr::match_template(input),
        M::CrossCorrelationNormalized => methods::CcorrNormalized::match_template(input),
    }
}

#[cfg(feature = "rayon")]
#[doc = generate_parallel_doc_comment!("match_template")]
pub fn match_template_parallel(
    image: &GrayImage,
    template: &GrayImage,
    method: MatchTemplateMethod,
) -> Image<Luma<f32>> {
    use MatchTemplateMethod as M;

    let input = &ImageTemplate::new(image, template);
    match method {
        M::SumOfSquaredErrors => methods::Sse::match_template_parallel(input),
        M::SumOfSquaredErrorsNormalized => methods::SseNormalized::match_template_parallel(input),
        M::CrossCorrelation => methods::Ccorr::match_template_parallel(input),
        M::CrossCorrelationNormalized => methods::CcorrNormalized::match_template_parallel(input),
    }
}

/// Slides a `template` over an `image` and scores the match at each point using
/// the requested `method`, computing the correlation terms using FFTs.
///
/// This is equivalent to [`match_template`], but can be faster for large
/// templates. It uses more temporary memory, and the results may differ from
/// [`match_template`] by small floating point roundoff.
///
/// The returned image has dimensions `image.width() - template.width() + 1` by
/// `image.height() - template.height() + 1`.
///
/// See [`MatchTemplateMethod`] for details of the matching methods.
///
/// # Panics
///
/// - If either dimension of `template` is zero.
/// - If either dimension of `template` is greater than the corresponding dimension
///   of `image`.
#[cfg(feature = "fft")]
pub fn match_template_fft(
    image: &GrayImage,
    template: &GrayImage,
    method: MatchTemplateMethod,
) -> Image<Luma<f32>> {
    use MatchTemplateMethod as M;

    let input = &ImageTemplate::new(image, template);
    match method {
        M::SumOfSquaredErrors => methods::Sse::match_template_fft(input),
        M::SumOfSquaredErrorsNormalized => methods::SseNormalized::match_template_fft(input),
        M::CrossCorrelation => methods::Ccorr::match_template_fft(input),
        M::CrossCorrelationNormalized => methods::CcorrNormalized::match_template_fft(input),
    }
}

/// Slides a `template` and a `mask` over an `image` and scores the match at each point using
/// the requested `method`.
///
/// The returned image has dimensions `image.width() - template.width() + 1` by
/// `image.height() - template.height() + 1`.
///
/// See [`MatchTemplateMethod`] for details of the matching methods.
///
/// # Panics
///
/// - If either dimension of `template` is not strictly less than the corresponding dimension
///   of `image`.
/// - If `template.dimensions() != mask.dimensions()`.
pub fn match_template_with_mask(
    image: &GrayImage,
    template: &GrayImage,
    method: MatchTemplateMethod,
    mask: &GrayImage,
) -> Image<Luma<f32>> {
    use MatchTemplateMethod as M;

    let input = &ImageTemplateMask::new(image, template, mask);
    match method {
        M::SumOfSquaredErrors => methods::SseWithMask::match_template(input),
        M::SumOfSquaredErrorsNormalized => methods::SseNormalizedWithMask::match_template(input),
        M::CrossCorrelation => methods::CcorrWithMask::match_template(input),
        M::CrossCorrelationNormalized => methods::CcorrNormalizedWithMask::match_template(input),
    }
}

#[cfg(feature = "rayon")]
#[doc = generate_parallel_doc_comment!("match_template_with_mask")]
pub fn match_template_with_mask_parallel(
    image: &GrayImage,
    template: &GrayImage,
    method: MatchTemplateMethod,
    mask: &GrayImage,
) -> Image<Luma<f32>> {
    use MatchTemplateMethod as M;

    let input = &ImageTemplateMask::new(image, template, mask);
    match method {
        M::SumOfSquaredErrors => methods::SseWithMask::match_template_parallel(input),
        M::SumOfSquaredErrorsNormalized => {
            methods::SseNormalizedWithMask::match_template_parallel(input)
        }
        M::CrossCorrelation => methods::CcorrWithMask::match_template_parallel(input),
        M::CrossCorrelationNormalized => {
            methods::CcorrNormalizedWithMask::match_template_parallel(input)
        }
    }
}

/// Slides a `template` and a `mask` over an `image` and scores the match at each
/// point using the requested `method`, computing the correlation terms using
/// FFTs.
///
/// This is equivalent to [`match_template_with_mask`], but can be faster for
/// large templates. It uses more temporary memory, and the results may differ
/// from [`match_template_with_mask`] by small floating point roundoff.
///
/// The returned image has dimensions `image.width() - template.width() + 1` by
/// `image.height() - template.height() + 1`.
///
/// See [`MatchTemplateMethod`] for details of the matching methods.
///
/// # Panics
///
/// - If either dimension of `template` is zero.
/// - If either dimension of `template` is greater than the corresponding
///   dimension of `image`.
/// - If `template.dimensions() != mask.dimensions()`.
#[cfg(feature = "fft")]
pub fn match_template_with_mask_fft(
    image: &GrayImage,
    template: &GrayImage,
    method: MatchTemplateMethod,
    mask: &GrayImage,
) -> Image<Luma<f32>> {
    use MatchTemplateMethod as M;

    let input = &ImageTemplateMask::new(image, template, mask);
    match method {
        M::SumOfSquaredErrors => methods::SseWithMask::match_template_fft(input),
        M::SumOfSquaredErrorsNormalized => {
            methods::SseNormalizedWithMask::match_template_fft(input)
        }
        M::CrossCorrelation => methods::CcorrWithMask::match_template_fft(input),
        M::CrossCorrelationNormalized => {
            methods::CcorrNormalizedWithMask::match_template_fft(input)
        }
    }
}

trait MatchTemplate<'a>
where
    Self: Sync + Sized,
{
    type Input: Sync + OutputDims;

    fn init(input: &Self::Input) -> Self;
    fn score_at(&self, at: (u32, u32), input: &Self::Input) -> f32;

    fn match_template(input: &Self::Input) -> Image<Luma<f32>> {
        let method = Self::init(input);
        let (width, height) = input.output_dims();

        Image::from_fn(width, height, |x, y| {
            let score = method.score_at((x, y), input);
            Luma([score])
        })
    }
    #[cfg(feature = "rayon")]
    fn match_template_parallel(input: &Self::Input) -> Image<Luma<f32>> {
        use rayon::prelude::*;

        let method = Self::init(input);
        let (width, height) = input.output_dims();

        let rows = (0..height)
            .into_par_iter()
            .map(|y| {
                (0..width)
                    .map(|x| method.score_at((x, y), input))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Image::from_fn(width, height, |x, y| {
            let score = rows[y as usize][x as usize];
            Luma([score])
        })
    }
}

#[cfg(feature = "fft")]
trait MatchTemplateFft<'a>
where
    Self: Sized,
{
    type Input: OutputDims;

    fn match_template_fft(input: &Self::Input) -> Image<Luma<f32>>;
}

trait OutputDims {
    fn output_dims(&self) -> (u32, u32);
}

mod methods {
    use super::*;

    pub struct Sse;
    impl<'a> MatchTemplate<'a> for Sse {
        type Input = ImageTemplate<'a>;
        fn init(_: &Self::Input) -> Self {
            Self
        }
        fn score_at(&self, at: (u32, u32), input: &Self::Input) -> f32 {
            let mut score = 0f32;
            unsafe {
                input.slide_window_at(at, |i, t| {
                    score += (t - i).powi(2);
                })
            };
            score
        }
    }

    pub struct SseNormalized {
        template_squared_sum: f32,
    }
    impl<'a> MatchTemplate<'a> for SseNormalized {
        type Input = ImageTemplate<'a>;
        fn init(input: &Self::Input) -> Self {
            Self {
                template_squared_sum: square_sum(input.template),
            }
        }
        fn score_at(&self, at: (u32, u32), input: &Self::Input) -> f32 {
            let mut score = 0f32;
            let mut ii = 0f32;
            unsafe {
                input.slide_window_at(at, |i, t| {
                    score += (t - i).powi(2);
                    ii += i * i;
                })
            };
            let norm = (ii * self.template_squared_sum).sqrt();
            if norm > 0.0 { score / norm } else { score }
        }
    }

    pub struct Ccorr;
    impl<'a> MatchTemplate<'a> for Ccorr {
        type Input = ImageTemplate<'a>;
        fn init(_: &Self::Input) -> Self {
            Self
        }
        fn score_at(&self, at: (u32, u32), input: &Self::Input) -> f32 {
            let mut score = 0f32;
            unsafe {
                input.slide_window_at(at, |i, t| {
                    score += i * t;
                })
            };
            score
        }
    }

    pub struct CcorrNormalized {
        template_squared_sum: f32,
    }
    impl<'a> MatchTemplate<'a> for CcorrNormalized {
        type Input = ImageTemplate<'a>;
        fn init(input: &Self::Input) -> Self {
            Self {
                template_squared_sum: square_sum(input.template),
            }
        }
        fn score_at(&self, at: (u32, u32), input: &Self::Input) -> f32 {
            let mut score = 0f32;
            let mut ii = 0f32;
            unsafe {
                input.slide_window_at(at, |i, t| {
                    score += i * t;
                    ii += i * i;
                })
            };
            let norm = (ii * self.template_squared_sum).sqrt();
            if norm > 0.0 { score / norm } else { score }
        }
    }

    pub struct SseWithMask;
    impl<'a> MatchTemplate<'a> for SseWithMask {
        type Input = ImageTemplateMask<'a>;
        fn init(_: &Self::Input) -> Self {
            Self
        }
        fn score_at(&self, at: (u32, u32), input: &Self::Input) -> f32 {
            let mut score = 0f32;
            unsafe {
                input.slide_window_at(at, |i, t, m| {
                    score += ((t - i) * m).powi(2);
                })
            };
            score
        }
    }

    pub struct SseNormalizedWithMask {
        template_mask_squared_sum: f32,
    }
    impl<'a> MatchTemplate<'a> for SseNormalizedWithMask {
        type Input = ImageTemplateMask<'a>;
        fn init(input: &Self::Input) -> Self {
            let template_mask_squared_sum = mult_square_sum(input.inner.template, input.mask);
            Self {
                template_mask_squared_sum,
            }
        }
        fn score_at(&self, at: (u32, u32), input: &Self::Input) -> f32 {
            let mut score = 0f32;
            let mut im_im = 0f32;
            unsafe {
                input.slide_window_at(at, |i, t, m| {
                    score += ((t - i) * m).powi(2);
                    im_im += (i * m).powi(2);
                })
            };
            let norm = (self.template_mask_squared_sum * im_im).sqrt();
            if norm > 0.0 { score / norm } else { score }
        }
    }
    pub struct CcorrWithMask;
    impl<'a> MatchTemplate<'a> for CcorrWithMask {
        type Input = ImageTemplateMask<'a>;
        fn init(_: &Self::Input) -> Self {
            Self
        }
        fn score_at(&self, at: (u32, u32), input: &Self::Input) -> f32 {
            let mut score = 0f32;
            unsafe {
                input.slide_window_at(at, |i, t, m| {
                    score += t * i * m * m;
                })
            };
            score
        }
    }

    pub struct CcorrNormalizedWithMask {
        template_mask_squared_sum: f32,
    }
    impl<'a> MatchTemplate<'a> for CcorrNormalizedWithMask {
        type Input = ImageTemplateMask<'a>;
        fn init(input: &Self::Input) -> Self {
            let template_mask_squared_sum = mult_square_sum(input.inner.template, input.mask);
            Self {
                template_mask_squared_sum,
            }
        }
        fn score_at(&self, at: (u32, u32), input: &Self::Input) -> f32 {
            let mut score = 0f32;
            let mut im_im = 0f32;
            unsafe {
                input.slide_window_at(at, |i, t, m| {
                    score += t * i * m * m;
                    im_im += (i * m).powi(2);
                })
            };
            let norm = (self.template_mask_squared_sum * im_im).sqrt();
            if norm > 0.0 { score / norm } else { score }
        }
    }

    #[cfg(feature = "fft")]
    impl<'a> MatchTemplateFft<'a> for Sse {
        type Input = ImageTemplate<'a>;

        fn match_template_fft(input: &Self::Input) -> Image<Luma<f32>> {
            fft::match_template(input, fft::ScoreKind::Sse)
        }
    }

    #[cfg(feature = "fft")]
    impl<'a> MatchTemplateFft<'a> for SseNormalized {
        type Input = ImageTemplate<'a>;

        fn match_template_fft(input: &Self::Input) -> Image<Luma<f32>> {
            fft::match_template(input, fft::ScoreKind::SseNormalized)
        }
    }

    #[cfg(feature = "fft")]
    impl<'a> MatchTemplateFft<'a> for Ccorr {
        type Input = ImageTemplate<'a>;

        fn match_template_fft(input: &Self::Input) -> Image<Luma<f32>> {
            fft::match_template(input, fft::ScoreKind::Ccorr)
        }
    }

    #[cfg(feature = "fft")]
    impl<'a> MatchTemplateFft<'a> for CcorrNormalized {
        type Input = ImageTemplate<'a>;

        fn match_template_fft(input: &Self::Input) -> Image<Luma<f32>> {
            fft::match_template(input, fft::ScoreKind::CcorrNormalized)
        }
    }

    #[cfg(feature = "fft")]
    impl<'a> MatchTemplateFft<'a> for SseWithMask {
        type Input = ImageTemplateMask<'a>;

        fn match_template_fft(input: &Self::Input) -> Image<Luma<f32>> {
            fft::match_template_with_mask(input, fft::ScoreKind::Sse)
        }
    }

    #[cfg(feature = "fft")]
    impl<'a> MatchTemplateFft<'a> for SseNormalizedWithMask {
        type Input = ImageTemplateMask<'a>;

        fn match_template_fft(input: &Self::Input) -> Image<Luma<f32>> {
            fft::match_template_with_mask(input, fft::ScoreKind::SseNormalized)
        }
    }

    #[cfg(feature = "fft")]
    impl<'a> MatchTemplateFft<'a> for CcorrWithMask {
        type Input = ImageTemplateMask<'a>;

        fn match_template_fft(input: &Self::Input) -> Image<Luma<f32>> {
            fft::match_template_with_mask(input, fft::ScoreKind::Ccorr)
        }
    }

    #[cfg(feature = "fft")]
    impl<'a> MatchTemplateFft<'a> for CcorrNormalizedWithMask {
        type Input = ImageTemplateMask<'a>;

        fn match_template_fft(input: &Self::Input) -> Image<Luma<f32>> {
            fft::match_template_with_mask(input, fft::ScoreKind::CcorrNormalized)
        }
    }

    fn square_sum(input: &GrayImage) -> f32 {
        input.iter().map(|&x| x as f32 * x as f32).sum()
    }
    fn mult_square_sum(a: &GrayImage, b: &GrayImage) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as f32 * y as f32).powi(2))
            .sum()
    }
}

struct ImageTemplate<'a> {
    image: &'a GrayImage,
    template: &'a GrayImage,
}
impl<'a> ImageTemplate<'a> {
    fn new(image: &'a GrayImage, template: &'a GrayImage) -> Self {
        assert!(
            image.width() >= template.width(),
            "image width must be greater than or equal to template width"
        );
        assert!(
            image.height() >= template.height(),
            "image height must be greater than or equal to template height"
        );
        Self { image, template }
    }
    unsafe fn slide_window_at(&self, (x, y): (u32, u32), mut for_each: impl FnMut(f32, f32)) {
        let (image, template) = (self.image, self.template);
        debug_assert!(x + template.width() - 1 < image.width());
        debug_assert!(y + template.height() - 1 < image.height());

        for dy in 0..template.height() {
            for dx in 0..template.width() {
                let image_value = unsafe { image.unsafe_get_pixel(x + dx, y + dy)[0] as f32 };
                let template_value = unsafe { template.unsafe_get_pixel(dx, dy)[0] as f32 };
                for_each(image_value, template_value);
            }
        }
    }
}
impl OutputDims for ImageTemplate<'_> {
    fn output_dims(&self) -> (u32, u32) {
        let width = self.image.width() - self.template.width() + 1;
        let height = self.image.height() - self.template.height() + 1;
        (width, height)
    }
}

struct ImageTemplateMask<'a> {
    inner: ImageTemplate<'a>,
    mask: &'a GrayImage,
}
impl<'a> ImageTemplateMask<'a> {
    fn new(image: &'a GrayImage, template: &'a GrayImage, mask: &'a GrayImage) -> Self {
        assert_eq!(
            template.dimensions(),
            mask.dimensions(),
            "the template and mask must be the same size"
        );
        let inner = ImageTemplate::new(image, template);
        Self { inner, mask }
    }
    unsafe fn slide_window_at(&self, (x, y): (u32, u32), mut for_each: impl FnMut(f32, f32, f32)) {
        let Self { mask, inner } = self;
        let (image, template) = (inner.image, inner.template);
        debug_assert!(x + template.width() - 1 < image.width());
        debug_assert!(y + template.height() - 1 < image.height());

        for dy in 0..template.height() {
            for dx in 0..template.width() {
                let image_value = unsafe { image.unsafe_get_pixel(x + dx, y + dy)[0] as f32 };
                let template_value = unsafe { template.unsafe_get_pixel(dx, dy)[0] as f32 };
                let mask_value = unsafe { mask.unsafe_get_pixel(dx, dy)[0] as f32 };
                for_each(image_value, template_value, mask_value);
            }
        }
    }
}
impl OutputDims for ImageTemplateMask<'_> {
    fn output_dims(&self) -> (u32, u32) {
        self.inner.output_dims()
    }
}

#[cfg(feature = "fft")]
mod fft {
    use super::*;
    use crate::integral_image::{integral_squared_image, sum_image_pixels};
    use rustdct::rustfft::{FftDirection, FftPlanner, num_complex::Complex as RustfftComplex};

    type Complex = RustfftComplex<f64>;

    #[derive(Copy, Clone)]
    pub(super) enum ScoreKind {
        Sse,
        SseNormalized,
        Ccorr,
        CcorrNormalized,
    }

    pub(super) fn match_template(
        input: &ImageTemplate<'_>,
        score_kind: ScoreKind,
    ) -> Image<Luma<f32>> {
        assert_non_empty(input.template);

        let cross_correlation = cross_correlate(
            input.image.dimensions(),
            input.template.dimensions(),
            |x, y| input.image.get_pixel(x, y)[0] as f64,
            |x, y| input.template.get_pixel(x, y)[0] as f64,
        );

        let image_squared_sum = if score_kind.needs_image_squared_sum() {
            Some(image_square_sums(input))
        } else {
            None
        };

        Terms {
            output_dims: input.output_dims(),
            cross_correlation,
            image_squared_sum,
            template_squared_sum: square_sum(input.template),
        }
        .scores(score_kind)
    }

    pub(super) fn match_template_with_mask(
        input: &ImageTemplateMask<'_>,
        score_kind: ScoreKind,
    ) -> Image<Luma<f32>> {
        let image = input.inner.image;
        let template = input.inner.template;
        let mask = input.mask;

        assert_non_empty(template);

        let cross_correlation = cross_correlate(
            image.dimensions(),
            template.dimensions(),
            |x, y| image.get_pixel(x, y)[0] as f64,
            |x, y| {
                let template_value = template.get_pixel(x, y)[0] as f64;
                let mask_value = mask.get_pixel(x, y)[0] as f64;
                template_value * mask_value * mask_value
            },
        );

        let image_squared_sum = if score_kind.needs_image_squared_sum() {
            Some(cross_correlate(
                image.dimensions(),
                template.dimensions(),
                |x, y| {
                    let value = image.get_pixel(x, y)[0] as f64;
                    value * value
                },
                |x, y| {
                    let mask_value = mask.get_pixel(x, y)[0] as f64;
                    mask_value * mask_value
                },
            ))
        } else {
            None
        };

        Terms {
            output_dims: input.output_dims(),
            cross_correlation,
            image_squared_sum,
            template_squared_sum: mult_square_sum(template, mask),
        }
        .scores(score_kind)
    }

    struct Terms {
        output_dims: (u32, u32),
        cross_correlation: Vec<f64>,
        image_squared_sum: Option<Vec<f64>>,
        template_squared_sum: f64,
    }

    impl Terms {
        fn scores(self, score_kind: ScoreKind) -> Image<Luma<f32>> {
            let Terms {
                output_dims: (width, height),
                cross_correlation,
                image_squared_sum,
                template_squared_sum,
            } = self;

            match score_kind {
                ScoreKind::Ccorr => image_from_scores(width, height, |index| {
                    clean_roundoff(cross_correlation[index])
                }),
                ScoreKind::CcorrNormalized => {
                    let image_squared_sum =
                        image_squared_sum.expect("image squared sums are required");
                    image_from_scores(width, height, |index| {
                        let cross_correlation = clean_roundoff(cross_correlation[index]);
                        let image_squared_sum = nonnegative_roundoff(image_squared_sum[index]);
                        let norm = (image_squared_sum * template_squared_sum).sqrt();
                        if norm > 0.0 {
                            cross_correlation / norm
                        } else {
                            cross_correlation
                        }
                    })
                }
                ScoreKind::Sse => {
                    let image_squared_sum =
                        image_squared_sum.expect("image squared sums are required");
                    image_from_scores(width, height, |index| {
                        let cross_correlation = clean_roundoff(cross_correlation[index]);
                        let image_squared_sum = nonnegative_roundoff(image_squared_sum[index]);
                        sse_from_terms(template_squared_sum, image_squared_sum, cross_correlation)
                    })
                }
                ScoreKind::SseNormalized => {
                    let image_squared_sum =
                        image_squared_sum.expect("image squared sums are required");
                    image_from_scores(width, height, |index| {
                        let cross_correlation = clean_roundoff(cross_correlation[index]);
                        let image_squared_sum = nonnegative_roundoff(image_squared_sum[index]);
                        let score = sse_from_terms(
                            template_squared_sum,
                            image_squared_sum,
                            cross_correlation,
                        );
                        let norm = (image_squared_sum * template_squared_sum).sqrt();
                        if norm > 0.0 { score / norm } else { score }
                    })
                }
            }
        }
    }

    impl ScoreKind {
        fn needs_image_squared_sum(self) -> bool {
            !matches!(self, ScoreKind::Ccorr)
        }
    }

    fn assert_non_empty(template: &GrayImage) {
        assert!(
            template.width() > 0 && template.height() > 0,
            "template must be non-empty"
        );
    }

    fn sse_from_terms(
        template_squared_sum: f64,
        image_squared_sum: f64,
        cross_correlation: f64,
    ) -> f64 {
        let score = template_squared_sum + image_squared_sum - 2.0 * cross_correlation;
        let roundoff_tolerance = 1e-9
            * (template_squared_sum + image_squared_sum + 2.0 * cross_correlation.abs()).max(1.0);

        if score < 0.0 && score.abs() <= roundoff_tolerance {
            0.0
        } else {
            clean_roundoff(score)
        }
    }

    fn clean_roundoff(value: f64) -> f64 {
        if value.abs() <= 1e-9 { 0.0 } else { value }
    }

    fn nonnegative_roundoff(value: f64) -> f64 {
        let value = clean_roundoff(value);
        if value < 0.0 && value > -1e-9 {
            0.0
        } else {
            value
        }
    }

    fn image_from_scores(
        width: u32,
        height: u32,
        mut score_at: impl FnMut(usize) -> f64,
    ) -> Image<Luma<f32>> {
        let width_usize = usize::try_from(width).unwrap();

        Image::from_fn(width, height, |x, y| {
            let index = y as usize * width_usize + x as usize;
            Luma([score_at(index) as f32])
        })
    }

    fn square_sum(input: &GrayImage) -> f64 {
        input
            .iter()
            .map(|&x| {
                let x = f64::from(x);
                x * x
            })
            .sum()
    }

    fn mult_square_sum(a: &GrayImage, b: &GrayImage) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let x = f64::from(x);
                let y = f64::from(y);
                (x * y).powi(2)
            })
            .sum()
    }

    fn image_square_sums(input: &ImageTemplate<'_>) -> Vec<f64> {
        let integral = integral_squared_image::<_, f64>(input.image);
        let (output_width, output_height) = input.output_dims();
        let (template_width, template_height) = input.template.dimensions();

        let mut output = Vec::with_capacity(output_len(output_width, output_height));
        for y in 0..output_height {
            for x in 0..output_width {
                output.push(
                    sum_image_pixels(
                        &integral,
                        x,
                        y,
                        x + template_width - 1,
                        y + template_height - 1,
                    )[0],
                );
            }
        }

        output
    }

    fn cross_correlate(
        image_dims: (u32, u32),
        kernel_dims: (u32, u32),
        mut image_value: impl FnMut(u32, u32) -> f64,
        mut kernel_value: impl FnMut(u32, u32) -> f64,
    ) -> Vec<f64> {
        let (image_width, image_height) = image_dims;
        let (kernel_width, kernel_height) = kernel_dims;

        debug_assert!(kernel_width > 0);
        debug_assert!(kernel_height > 0);
        debug_assert!(image_width >= kernel_width);
        debug_assert!(image_height >= kernel_height);

        let output_width = image_width - kernel_width + 1;
        let output_height = image_height - kernel_height + 1;
        let convolution_width = usize::try_from(image_width)
            .unwrap()
            .checked_add(usize::try_from(kernel_width).unwrap())
            .and_then(|x| x.checked_sub(1))
            .unwrap();
        let convolution_height = usize::try_from(image_height)
            .unwrap()
            .checked_add(usize::try_from(kernel_height).unwrap())
            .and_then(|x| x.checked_sub(1))
            .unwrap();
        let fft_width = convolution_width
            .checked_next_power_of_two()
            .expect("FFT width is too large");
        let fft_height = convolution_height
            .checked_next_power_of_two()
            .expect("FFT height is too large");
        let fft_len = fft_width
            .checked_mul(fft_height)
            .expect("FFT buffer is too large");

        let zero = Complex::new(0.0, 0.0);
        let mut image_fft = vec![zero; fft_len];
        let mut kernel_fft = vec![zero; fft_len];

        for y in 0..image_height {
            let row_offset = y as usize * fft_width;
            for x in 0..image_width {
                image_fft[row_offset + x as usize].re = image_value(x, y);
            }
        }

        for y in 0..kernel_height {
            let flipped_y = (kernel_height - 1 - y) as usize;
            let row_offset = flipped_y * fft_width;
            for x in 0..kernel_width {
                let flipped_x = (kernel_width - 1 - x) as usize;
                kernel_fft[row_offset + flipped_x].re = kernel_value(x, y);
            }
        }

        let mut planner = FftPlanner::<f64>::new();
        fft2d(
            &mut image_fft,
            fft_width,
            fft_height,
            FftDirection::Forward,
            &mut planner,
        );
        fft2d(
            &mut kernel_fft,
            fft_width,
            fft_height,
            FftDirection::Forward,
            &mut planner,
        );

        for (image_frequency, kernel_frequency) in image_fft.iter_mut().zip(kernel_fft) {
            *image_frequency *= kernel_frequency;
        }

        fft2d(
            &mut image_fft,
            fft_width,
            fft_height,
            FftDirection::Inverse,
            &mut planner,
        );

        let scale = fft_len as f64;
        let mut output = Vec::with_capacity(output_len(output_width, output_height));
        for y in 0..output_height {
            let fft_y = y + kernel_height - 1;
            let row_offset = fft_y as usize * fft_width;
            for x in 0..output_width {
                let fft_x = x + kernel_width - 1;
                output.push(clean_roundoff(
                    image_fft[row_offset + fft_x as usize].re / scale,
                ));
            }
        }

        output
    }

    fn fft2d(
        values: &mut [Complex],
        width: usize,
        height: usize,
        direction: FftDirection,
        planner: &mut FftPlanner<f64>,
    ) {
        debug_assert_eq!(values.len(), width * height);

        let row_fft = planner.plan_fft(width, direction);
        let column_fft = planner.plan_fft(height, direction);

        let mut row_scratch = vec![Complex::new(0.0, 0.0); row_fft.get_inplace_scratch_len()];
        for row in values.chunks_exact_mut(width) {
            row_fft.process_with_scratch(row, &mut row_scratch);
        }

        let mut column = vec![Complex::new(0.0, 0.0); height];
        let mut column_scratch = vec![Complex::new(0.0, 0.0); column_fft.get_inplace_scratch_len()];
        for x in 0..width {
            for y in 0..height {
                column[y] = values[y * width + x];
            }
            column_fft.process_with_scratch(&mut column, &mut column_scratch);
            for y in 0..height {
                values[y * width + x] = column[y];
            }
        }
    }

    fn output_len(width: u32, height: u32) -> usize {
        usize::try_from(width)
            .unwrap()
            .checked_mul(usize::try_from(height).unwrap())
            .expect("output image is too large")
    }
}

/// The largest and smallest values in an image,
/// together with their locations.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Extremes<T> {
    /// The largest value in an image.
    pub max_value: T,
    /// The smallest value in an image.
    pub min_value: T,
    /// The coordinates of the largest value in an image.
    pub max_value_location: (u32, u32),
    /// The coordinates of the smallest value in an image.
    pub min_value_location: (u32, u32),
}

/// Finds the largest and smallest values in an image and their locations.
/// If there are multiple such values then the lexicographically smallest is returned.
pub fn find_extremes<T>(image: &Image<Luma<T>>) -> Extremes<T>
where
    T: Primitive,
{
    assert!(
        image.width() > 0 && image.height() > 0,
        "image must be non-empty"
    );

    let mut min_value = image.get_pixel(0, 0)[0];
    let mut max_value = image.get_pixel(0, 0)[0];

    let mut min_value_location = (0, 0);
    let mut max_value_location = (0, 0);

    for (x, y, p) in image.enumerate_pixels() {
        if p[0] < min_value {
            min_value = p[0];
            min_value_location = (x, y);
        }
        if p[0] > max_value {
            max_value = p[0];
            max_value_location = (x, y);
        }
    }

    Extremes {
        max_value,
        min_value,
        max_value_location,
        min_value_location,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;

    #[test]
    #[should_panic]
    fn match_template_panics_if_image_width_does_is_less_than_template_width() {
        let _ = match_template(
            &GrayImage::new(5, 5),
            &GrayImage::new(6, 5),
            MatchTemplateMethod::SumOfSquaredErrors,
        );
    }

    #[test]
    #[should_panic]
    fn match_template_panics_if_image_height_is_less_than_template_height() {
        let _ = match_template(
            &GrayImage::new(5, 5),
            &GrayImage::new(5, 6),
            MatchTemplateMethod::SumOfSquaredErrors,
        );
    }

    #[test]
    fn match_template_handles_template_of_same_size_as_image() {
        assert_pixels_eq!(
            match_template(
                &GrayImage::new(5, 5),
                &GrayImage::new(5, 5),
                MatchTemplateMethod::SumOfSquaredErrors
            ),
            gray_image!(type: f32, 0.0)
        );
    }

    #[test]
    fn match_template_normalization_handles_zero_norm() {
        assert_pixels_eq!(
            match_template(
                &GrayImage::new(1, 1),
                &GrayImage::new(1, 1),
                MatchTemplateMethod::SumOfSquaredErrorsNormalized
            ),
            gray_image!(type: f32, 0.0)
        );
    }

    #[cfg(feature = "fft")]
    #[test]
    #[should_panic(expected = "template must be non-empty")]
    fn match_template_fft_panics_if_template_is_empty() {
        let _ = match_template_fft(
            &GrayImage::new(5, 5),
            &GrayImage::new(0, 5),
            MatchTemplateMethod::SumOfSquaredErrors,
        );
    }

    #[cfg(feature = "fft")]
    #[test]
    fn match_template_fft_matches_match_template() {
        let image = gray_image!(
            1, 4, 2, 5;
            2, 1, 3, 6;
            3, 3, 4, 7;
            4, 1, 2, 8
        );
        let template = gray_image!(
            1, 2, 1;
            3, 4, 2
        );

        for method in [
            MatchTemplateMethod::SumOfSquaredErrors,
            MatchTemplateMethod::SumOfSquaredErrorsNormalized,
            MatchTemplateMethod::CrossCorrelation,
            MatchTemplateMethod::CrossCorrelationNormalized,
        ] {
            let expected = match_template(&image, &template, method);
            let actual = match_template_fft(&image, &template, method);

            assert_pixels_eq_within!(actual, expected, 1e-4f32);
        }
    }

    #[cfg(feature = "fft")]
    #[test]
    fn match_template_fft_normalization_handles_zero_norm() {
        let image = GrayImage::new(3, 3);
        let template = gray_image!(
            1, 0;
            0, 0
        );

        for method in [
            MatchTemplateMethod::SumOfSquaredErrorsNormalized,
            MatchTemplateMethod::CrossCorrelationNormalized,
        ] {
            let expected = match_template(&image, &template, method);
            let actual = match_template_fft(&image, &template, method);

            assert_pixels_eq_within!(actual, expected, 1e-4f32);
        }
    }

    #[cfg_attr(miri, ignore = "assert fails")]
    #[test]
    fn match_template_sum_of_squared_errors() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );

        let actual = match_template(&image, &template, MatchTemplateMethod::SumOfSquaredErrors);
        let expected = gray_image!(type: f32,
            14.0, 14.0;
            3.0, 1.0
        );

        assert_pixels_eq!(actual, expected);
    }

    #[cfg_attr(miri, ignore = "assert fails")]
    #[test]
    fn match_template_sum_of_squared_errors_normalized() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );

        let actual = match_template(
            &image,
            &template,
            MatchTemplateMethod::SumOfSquaredErrorsNormalized,
        );
        let tss = 30f32;
        let expected = gray_image!(type: f32,
            14.0 / (22.0 * tss).sqrt(), 14.0 / (30.0 * tss).sqrt();
            3.0 / (23.0 * tss).sqrt(), 1.0 / (35.0 * tss).sqrt()
        );

        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn match_template_cross_correlation() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );

        let actual = match_template(&image, &template, MatchTemplateMethod::CrossCorrelation);
        let expected = gray_image!(type: f32,
            19.0, 23.0;
            25.0, 32.0
        );

        assert_pixels_eq!(actual, expected);
    }

    #[cfg_attr(miri, ignore = "assert fails")]
    #[test]
    fn match_template_cross_correlation_normalized() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );

        let actual = match_template(
            &image,
            &template,
            MatchTemplateMethod::CrossCorrelationNormalized,
        );
        let tss = 30f32;
        let expected = gray_image!(type: f32,
            19.0 / (22.0 * tss).sqrt(), 23.0 / (30.0 * tss).sqrt();
            25.0 / (23.0 * tss).sqrt(), 32.0 / (35.0 * tss).sqrt()
        );

        assert_pixels_eq!(actual, expected);
    }

    #[cfg_attr(miri, ignore = "assert fails")]
    #[test]
    fn match_template_sum_of_squared_errors_with_mask() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );
        let mask = gray_image!(
            0, 1;
            2, 3
        );
        let expected = gray_image!(type: f32,
            89., 25.;
            10., 1.
        );
        let actual = match_template_with_mask(
            &image,
            &template,
            MatchTemplateMethod::SumOfSquaredErrors,
            &mask,
        );
        assert_pixels_eq!(actual, expected);

        #[cfg(feature = "rayon")]
        {
            let actual_parallel = match_template_with_mask_parallel(
                &image,
                &template,
                MatchTemplateMethod::SumOfSquaredErrors,
                &mask,
            );
            assert_pixels_eq!(actual_parallel, expected);
        }
    }

    #[cfg_attr(miri, ignore = "assert fails")]
    #[test]
    fn match_template_sum_of_squared_errors_normalized_with_mask() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );
        let mask = gray_image!(
            0, 1;
            2, 3
        );
        let expected = gray_image!(type: f32,
            1.0246822 , 0.19536021;
            0.067865655, 0.005362412
        );
        let actual = match_template_with_mask(
            &image,
            &template,
            MatchTemplateMethod::SumOfSquaredErrorsNormalized,
            &mask,
        );
        assert_pixels_eq!(actual, expected);

        #[cfg(feature = "rayon")]
        {
            let actual_parallel = match_template_with_mask_parallel(
                &image,
                &template,
                MatchTemplateMethod::SumOfSquaredErrorsNormalized,
                &mask,
            );
            assert_pixels_eq!(actual_parallel, expected);
        }
    }

    #[test]
    fn match_template_cross_correlation_with_mask() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );
        let mask = gray_image!(
            0, 1;
            2, 3
        );
        let expected = gray_image!(type: f32,
            68., 124.;
            146., 186.
        );
        let actual = match_template_with_mask(
            &image,
            &template,
            MatchTemplateMethod::CrossCorrelation,
            &mask,
        );
        assert_pixels_eq!(actual, expected);

        #[cfg(feature = "rayon")]
        {
            let actual_parallel = match_template_with_mask_parallel(
                &image,
                &template,
                MatchTemplateMethod::CrossCorrelation,
                &mask,
            );
            assert_pixels_eq!(actual_parallel, expected);
        }
    }

    #[cfg_attr(miri, ignore = "assert fails")]
    #[test]
    fn match_template_cross_correlation_normalized_with_mask() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );
        let mask = gray_image!(
            0, 1;
            2, 3
        );
        let expected = gray_image!(type: f32,
            0.78290325, 0.96898663;
            0.9908386, 0.9974086
        );
        let actual = match_template_with_mask(
            &image,
            &template,
            MatchTemplateMethod::CrossCorrelationNormalized,
            &mask,
        );
        assert_pixels_eq!(actual, expected);

        #[cfg(feature = "rayon")]
        {
            let actual_parallel = match_template_with_mask_parallel(
                &image,
                &template,
                MatchTemplateMethod::CrossCorrelationNormalized,
                &mask,
            );
            assert_pixels_eq!(actual_parallel, expected);
        }
    }

    #[cfg(feature = "fft")]
    #[test]
    fn match_template_with_mask_fft_matches_match_template_with_mask() {
        let image = gray_image!(
            1, 4, 2, 5;
            2, 1, 3, 6;
            3, 3, 4, 7;
            4, 1, 2, 8
        );
        let template = gray_image!(
            1, 2, 1;
            3, 4, 2
        );
        let mask = gray_image!(
            0, 1, 2;
            3, 1, 0
        );

        for method in [
            MatchTemplateMethod::SumOfSquaredErrors,
            MatchTemplateMethod::SumOfSquaredErrorsNormalized,
            MatchTemplateMethod::CrossCorrelation,
            MatchTemplateMethod::CrossCorrelationNormalized,
        ] {
            let expected = match_template_with_mask(&image, &template, method, &mask);
            let actual = match_template_with_mask_fft(&image, &template, method, &mask);

            assert_pixels_eq_within!(actual, expected, 1e-4f32);
        }
    }

    #[cfg(feature = "fft")]
    #[test]
    fn match_template_with_mask_fft_normalization_handles_zero_norm() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );
        let mask = GrayImage::new(2, 2);

        for method in [
            MatchTemplateMethod::SumOfSquaredErrorsNormalized,
            MatchTemplateMethod::CrossCorrelationNormalized,
        ] {
            let expected = match_template_with_mask(&image, &template, method, &mask);
            let actual = match_template_with_mask_fft(&image, &template, method, &mask);

            assert_pixels_eq_within!(actual, expected, 1e-4f32);
        }
    }

    #[cfg(feature = "fft")]
    #[test]
    #[should_panic(expected = "template must be non-empty")]
    fn match_template_with_mask_fft_panics_if_template_is_empty() {
        let _ = match_template_with_mask_fft(
            &GrayImage::new(5, 5),
            &GrayImage::new(0, 5),
            MatchTemplateMethod::SumOfSquaredErrors,
            &GrayImage::new(0, 5),
        );
    }

    #[test]
    fn test_find_extremes() {
        let image = gray_image!(
            10,  7,  8,  1;
             9, 15,  4,  2
        );

        let expected = Extremes {
            max_value: 15,
            min_value: 1,
            max_value_location: (1, 1),
            min_value_location: (3, 0),
        };

        assert_eq!(find_extremes(&image), expected);
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod benches {
    use super::*;
    use crate::utils::gray_bench_image;
    use test::{Bencher, black_box};

    macro_rules! bench_match_template {
        ($name:ident, image_size: $s:expr, template_size: $t:expr, method: $m:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let image = gray_bench_image($s, $s);
                let template = gray_bench_image($t, $t);
                b.iter(|| {
                    let result =
                        match_template(&image, &template, MatchTemplateMethod::SumOfSquaredErrors);
                    black_box(result);
                })
            }
        };
    }

    bench_match_template!(
        bench_match_template_s100_t1_sse,
        image_size: 100,
        template_size: 1,
        method: MatchTemplateMethod::SumOfSquaredErrors);

    bench_match_template!(
        bench_match_template_s100_t4_sse,
        image_size: 100,
        template_size: 4,
        method: MatchTemplateMethod::SumOfSquaredErrors);

    bench_match_template!(
        bench_match_template_s100_t16_sse,
        image_size: 100,
        template_size: 16,
        method: MatchTemplateMethod::SumOfSquaredErrors);

    bench_match_template!(
        bench_match_template_s100_t1_sse_norm,
        image_size: 100,
        template_size: 1,
        method: MatchTemplateMethod::SumOfSquaredErrorsNormalized);

    bench_match_template!(
        bench_match_template_s100_t4_sse_norm,
        image_size: 100,
        template_size: 4,
        method: MatchTemplateMethod::SumOfSquaredErrorsNormalized);

    bench_match_template!(
        bench_match_template_s100_t16_sse_norm,
        image_size: 100,
        template_size: 16,
        method: MatchTemplateMethod::SumOfSquaredErrorsNormalized);

    macro_rules! bench_match_template_with_mask {
        ($name:ident, image_size: $s:expr, template_size: $t:expr, method: $m:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let image = gray_bench_image($s, $s);
                let template = gray_bench_image($t, $t);
                let mask = gray_bench_image($t, $t);
                b.iter(|| {
                    let result = match_template_with_mask(
                        &image,
                        &template,
                        MatchTemplateMethod::SumOfSquaredErrors,
                        &mask,
                    );
                    black_box(result);
                })
            }
        };
    }

    bench_match_template_with_mask!(
        bench_match_template_with_mask_s100_t1_sse,
        image_size: 100,
        template_size: 1,
        method: MatchTemplateMethod::SumOfSquaredErrors);

    bench_match_template_with_mask!(
        bench_match_template_with_mask_s100_t4_sse,
        image_size: 100,
        template_size: 4,
        method: MatchTemplateMethod::SumOfSquaredErrors);

    bench_match_template_with_mask!(
        bench_match_template_with_mask_s100_t16_sse,
        image_size: 100,
        template_size: 16,
        method: MatchTemplateMethod::SumOfSquaredErrors);

    bench_match_template_with_mask!(
        bench_match_template_with_mask_s100_t1_sse_norm,
        image_size: 100,
        template_size: 1,
        method: MatchTemplateMethod::SumOfSquaredErrorsNormalized);

    bench_match_template_with_mask!(
        bench_match_template_with_mask_s100_t4_sse_norm,
        image_size: 100,
        template_size: 4,
        method: MatchTemplateMethod::SumOfSquaredErrorsNormalized);

    bench_match_template_with_mask!(
        bench_match_template_with_mask_s100_t16_sse_norm,
        image_size: 100,
        template_size: 16,
        method: MatchTemplateMethod::SumOfSquaredErrorsNormalized);
}
