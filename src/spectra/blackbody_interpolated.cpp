#include <mitsuba/core/distr_1d.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _spectrum-blackbody_interpolated:

Black body radiance spectrum (:monosp:`blackbody_interpolated`)
---------------------------------------------------------------

.. pluginparameters::

    * - temperature
      - |float|
      - Black body temperature (in K).

This plugin computes the spectral radiance (in W/m²/sr/nm) in emitted by a black
body at the specified temperature (in K).

This implementation relies on a piecewise-linear discretisation in a
dimensionless wavelength space to mitigate the effects temperature can have on
the spectral profile's curvature.

*/

/**
 * NOTE: spectral variable naming convention
 *       - wavelength -> nm
 *       - lambda -> m
 */
template <typename Float, typename Spectrum>
class BlackBodyInterpolatedSpectrum final : public Texture<Float, Spectrum> {
public:
    MTS_IMPORT_TYPES(Texture)

    using FloatStorage = DynamicBuffer<Float>;

    // Basic physical constants
    constexpr static ScalarFloat c =
        ScalarFloat(2.99792458e+8); // Speed of light
    constexpr static ScalarFloat h =
        ScalarFloat(6.62607004e-34); // Planck constant
    constexpr static ScalarFloat k =
        ScalarFloat(1.38064852e-23); // Boltzmann constant

    // Other constants
    constexpr static ScalarFloat b =
        ScalarFloat(2.89777196e-3); // Wien's displacement constant [m.K]

    // Radiation constants
    // [https://en.wikipedia.org/wiki/Planck%27s_law#First_and_second_radiation_constants]
    constexpr static ScalarFloat c1 = 2 * h * c * c;
    constexpr static ScalarFloat c2 = h * c / k;

    /**
     * \brief Helper function evaluating Planck's function
     *
     * \param lambda Wavelength at which the Planck function is evaluated
     *      (in m)
     * \param temperature Temperature for which the Planck function is evaluated
     *      (in K)
     * \return Planck's function evaluation
     */
    static ScalarFloat planck(const ScalarFloat &lambda,
                              const ScalarFloat &temperature) {
        ScalarFloat lambda2 = sqr(lambda), lambda5 = sqr(lambda2) * lambda;
        return c1 / (lambda5 * (exp(c2 / (lambda * temperature)) - 1.f));
    }

    BlackBodyInterpolatedSpectrum(const Properties &props) : Texture(props) {
        m_temperature = props.float_("temperature");
        update();
    }

    void update() {
        m_peak_lambda = b / m_temperature;

        /* Populate the Planck distribution on dimensionless spectral space.
         * Discretisation on the dimensionless spectral space ensures good
         * accuracy regarless of temperature.
         */
        ScalarFloat x_min = MTS_WAVELENGTH_MIN * 1e-9f / m_peak_lambda,
                    x_max = MTS_WAVELENGTH_MAX * 1e-9f / m_peak_lambda;
        int n_x = int((x_max - x_min) * 1000 + 1); // One point per 1e-3 x
                                                   // usually ensures more than
                                                   // acceptable accuracy
        auto x = linspace<FloatStorage>(
            x_min, x_max, n_x); // Dimensionless wavelength space coordinates
        auto lambda =
            x * m_peak_lambda; // Corresponding wavelength space coordinates

        auto planck_values = zero<FloatStorage>(n_x);
        for (int i = 0; i != n_x; ++i)
            planck_values[i] = planck(lambda[i], m_temperature);

        m_planck.range() = ScalarVector2f(x_min, x_max);
        m_planck.pdf()   = planck_values;
        m_planck.update();
    }

    UnpolarizedSpectrum eval(const SurfaceInteraction3f &si,
                             Mask active) const override {
        if constexpr (is_spectral_v<Spectrum>) {
            Wavelength x = si.wavelengths * 1e-9f / m_peak_lambda;
            return m_planck.eval_pdf(x, active);
        } else {
            Throw("Not implemented for non-spectral modes");
        }
    }

    Wavelength pdf(const SurfaceInteraction3f &si, Mask active) const override {
        if constexpr (is_spectral_v<Spectrum>) {
            Wavelength x = si.wavelengths * 1e-9f / m_peak_lambda;
            return m_planck.eval_pdf_normalized(x, active);
        } else {
            Throw("Not implemented for non-spectral modes");
        }
    }

    std::pair<Wavelength, UnpolarizedSpectrum>
    sample(const SurfaceInteraction3f & /*si*/, const Wavelength &sample,
           Mask active) const override {
        if constexpr (is_spectral_v<Spectrum>) {
            return { m_planck.sample(sample, active), m_planck.integral() };
        } else {
            Throw("Not implemented for non-spectral modes");
        }
    }

    ScalarFloat mean() const override {
        return m_planck.integral() / (MTS_WAVELENGTH_MAX - MTS_WAVELENGTH_MIN);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("temperature", m_temperature);
    }

    void parameters_changed() override { update(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "BlackBodyInterpolatedSpectrum[" << std::endl
            << "  temperature = " << m_temperature << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ScalarFloat m_temperature; // Black body temperature
    ScalarFloat m_peak_lambda; // Wavelength corresponding to the maximum of
                               // Planck's function (in m)
    ContinuousDistribution<Wavelength> m_planck; // Planck function discretised
                                                 // in the dimensionless
                                                 // spectral space
};

MTS_IMPLEMENT_CLASS_VARIANT(BlackBodyInterpolatedSpectrum, Texture)
MTS_EXPORT_PLUGIN(BlackBodyInterpolatedSpectrum,
                  "Black body spectrum [interpolated]")
NAMESPACE_END(mitsuba)
