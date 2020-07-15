#include <mitsuba/core/distr_1d.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _spectrum-discrete:

Discrete spectrum (:monosp:`discrete`)
--------------------------------------

.. pluginparameters::

 * - wavelengths
   - |string|
   - A comma-separated list of wavelengths to sample.

 * - values
   - |string|
   - A comma-separated list of spectrum values associated with each wavelength. 
     Alternatively, a single value can be passed and used for all wavelengths.
     (Default: "1")

 * - pmf
   - |string|
   - A comma-separated list of probability mass density associated with each
     wavelength. If unspecified, all wavelengths are equiprobable.

*This spectrum can only be used through its full XML specification.*

This spectrum plugin samples wavelengths from a discrete distribution. 
Consequently, it will always return 0 when queried for evaluation or PDF values.

*/
template <typename Float, typename Spectrum>
class DiscreteSpectrum final : public Texture<Float, Spectrum> {
public:
    MTS_IMPORT_TYPES(Texture)

private:
    using FloatStorage = DynamicBuffer<Float>;

public:
    DiscreteSpectrum(const Properties &props) : Texture(props) {
        // Wavelengths are required
        std::vector<std::string> wavelengths_str =
            string::tokenize(props.string("wavelengths"), " ,");

        // Values are optional
        std::vector<std::string> values_str =
            string::tokenize(props.string("values", "1"), " ,");

        // If a single value is provided, use it for every wavelength
        if (values_str.size() == 1)
            for (size_t i = 1; i < wavelengths_str.size(); ++i)
                values_str.push_back(values_str[0]);

        // Check value vector size
        if (wavelengths_str.size() != values_str.size())
            Throw("DiscreteSpectrum: 'wavelengths' and 'values' parameters "
                  "must have the same size!");

        // PMF values are optional
        std::vector<std::string> pmf_str =
            string::tokenize(props.string("pmf", "1"), " ,");

        // If a single PMF value is provided, use it for every wavelength
        if (pmf_str.size() == 1)
            for (size_t i = 1; i < wavelengths_str.size(); ++i)
                pmf_str.push_back(pmf_str[0]);

        // Check PMF vector size
        if (wavelengths_str.size() != pmf_str.size())
            Throw("DiscreteSpectrum: 'wavelengths' and 'pmf' parameters "
                  "must have the same size!");

        m_wavelengths    = empty<FloatStorage>(wavelengths_str.size());
        m_values         = empty<FloatStorage>(wavelengths_str.size());
        FloatStorage pmf = empty<FloatStorage>(wavelengths_str.size());
        m_wavelengths.managed();
        m_values.managed();
        pmf.managed();

        for (size_t i = 0; i < wavelengths_str.size(); ++i) {
            try {
                m_wavelengths[i] = (ScalarFloat) std::stod(wavelengths_str[i]);
                m_values[i]      = (ScalarFloat) std::stod(values_str[i]);
                pmf[i]           = (ScalarFloat) std::stod(pmf_str[i]);
            } catch (...) {
                Throw("Could not parse floating point value '%s'",
                      wavelengths_str[i]);
            }
        }

        m_distr = DiscreteDistribution<Wavelength>(pmf);
    }

    void traverse(TraversalCallback * /*callback*/) override {
        // TODO: implement
        NotImplementedError("traverse");
    }

    void
    parameters_changed(const std::vector<std::string> & /*keys*/) override {
        // TODO: implement
        NotImplementedError("parameters_changed");
    }

    UnpolarizedSpectrum eval(const SurfaceInteraction3f & /*si*/,
                             Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return 0.f;
    }

    Wavelength pdf_spectrum(const SurfaceInteraction3f & /*si*/,
                            Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return 0.f;
    }

    std::pair<Wavelength, UnpolarizedSpectrum>
    sample_spectrum(const SurfaceInteraction3f & /*si*/,
                    const Wavelength &sample, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::TextureSample, active);

        if constexpr (is_spectral_v<Spectrum>) {
            auto index = m_distr.sample(sample, active);
            return { gather<Wavelength>(m_wavelengths, index, active),
                     gather<Wavelength>(m_values, index, active) };
        } else {
            ENOKI_MARK_USED(sample);
            NotImplementedError("sample_spectrum");
        }
    }

    ScalarFloat mean() const override { return 0.f; }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "DiscreteSpectrum[" << std::endl
            << "  wavelengths = " << m_wavelengths << std::endl
            << "  values = " << m_values << std::endl
            << "  distr = " << string::indent(m_distr) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    FloatStorage m_wavelengths;
    FloatStorage m_values;
    DiscreteDistribution<Wavelength> m_distr;
};

MTS_IMPLEMENT_CLASS_VARIANT(DiscreteSpectrum, Texture)
MTS_EXPORT_PLUGIN(DiscreteSpectrum, "Discrete spectrum")
NAMESPACE_END(mitsuba)
