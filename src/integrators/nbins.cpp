#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)
/**!

.. _integrator-nbins:

Narrow bins integrator (:monosp:`nbins`)
----------------------------------------

.. pluginparameters::

 * - wavelengths
   - |string|
   - A comma-separated list of wavelengths used for bin detection.

 * - tolerance
   - |float|
   - Tolerance for bin detection (in nm), *i.e.* narrow bin width.
     (Default: 1e-5)

 * - (Nested plugin)
   - :paramtype:`integrator`
   - Sub-integrator (only one can be specified) which will be sampled along the
     narrow bins integrator.

This integrator computes radiance for selected wavelengths in spectral mode.
It is intended to be used when wavelengths are sampled from discrete spectral
distributions.

In practice, it accumulates contributions in very narrow spectral bins whose
width is controlled by the ``tolerance`` parameter. In addition to accumulated
contributions for each wavelength, the integrator records the number of
contributions for each wavelength, which should be used during a post-processing
step to normalize the obtained values.

.. note::

   This integrator can only be used with non-polarized spectral variants.

.. warning::

   If used improperly, this integrator is very likely to yield meaningless
   results!

 */
template <typename Float, typename Spectrum>
class NarrowBinsIntegrator final : public SamplingIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(SamplingIntegrator)
    MTS_IMPORT_TYPES(Scene, Sensor, Sampler, Medium, Texture)

    NarrowBinsIntegrator(const Properties &props) : Base(props) {
        // If used in nonspectral mode, raise
        if constexpr (!is_spectral_v<Spectrum>)
            Throw("This integrator can only be used with a spectral variant!");

        // If used in polarized mode, raise
        if constexpr (is_polarized_v<Spectrum>)
            Throw("This integrator cannot (yet) be used in polarized mode!");

        // Get sub-integrator
        for (auto &kv : props.objects()) {
            Base *integrator = dynamic_cast<Base *>(kv.second.get());
            if (!integrator)
                Throw("Child objects must be of type 'SamplingIntegrator'!");
            if (m_integrator)
                Throw("More than one sub-integrator specified!");
            m_integrator = integrator;
        }

        if (!m_integrator)
            Throw("Must specify a sub-integrator!");

        // Parse wavelengths
        std::vector<std::string> tokens =
            string::tokenize(props.string("wavelengths"), " ,");

        for (const std::string &token : tokens) {
            try {
                m_bin_wavelengths.push_back((ScalarFloat) std::stod(token));
            } catch (...) {
                Throw("Could not parse floating point value '%s'", token);
            }
            m_bin_names.push_back(token);
        }

        if (m_bin_names.empty())
            Log(Warn, "No spectral bin was specfied!");

        // Get tolerance value
        m_tolerance = props.float_("tolerance", 1e-5);
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene, const Sensor *sensor,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray,
                                     const Medium *medium, Float *aovs,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        auto result =
            m_integrator->sample(scene, sensor, sampler, ray, medium, aovs, active);

        if constexpr (is_spectral_v<Spectrum>) {
            // Compute bin values
            for (size_t i = 0; i < m_bin_names.size(); ++i) {
                // Select wavelengths fitting current bin
                auto bin_mask =
                    abs(ray.wavelengths - m_bin_wavelengths[i]) <= m_tolerance;

                // Gather radiance values
                auto bin_values = select(bin_mask, result.first, 0.f);
                *aovs++         = hsum(bin_values);

                // Compute bin population for post-processing
                auto bin_population = select(bin_mask, Spectrum(1.f), 0.f);
                *aovs++             = hsum(bin_population);
            }
        }

        // Forward result
        return result;
    }

    std::vector<std::string> aov_names() const override {
        std::vector<std::string> names;
        for (auto &bin_name : m_bin_names) {
            names.push_back(bin_name);
            names.push_back(bin_name + "_pop");
        }
        return names;
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("integrator", m_integrator.get());
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "NarrowBinsIntegrator[" << std::endl
            << "  wavelengths = " << string::indent(m_bin_wavelengths) << ","
            << std::endl
            << "  integrator = " << string::indent(m_integrator) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    std::vector<std::string> m_bin_names;
    std::vector<ScalarFloat> m_bin_wavelengths;
    ScalarFloat m_tolerance;
    ref<Base> m_integrator;
};

MTS_IMPLEMENT_CLASS_VARIANT(NarrowBinsIntegrator, SamplingIntegrator)
MTS_EXPORT_PLUGIN(NarrowBinsIntegrator, "Narrow bins integrator");
NAMESPACE_END(mitsuba)
