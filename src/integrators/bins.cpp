#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

// This is still WIP. The basic idea is to implement a spectral
// equivalent to reconstruction filters.
// TODO: fix and finish this
// TODO: write docs
template <typename Float, typename Spectrum>
class BinsIntegrator final : public SamplingIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(SamplingIntegrator)
    MTS_IMPORT_TYPES(Scene, Sensor, Sampler, Medium, Texture)

    BinsIntegrator(const Properties &props) : Base(props) {
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

        // Parse bin specification
        std::vector<std::string> tokens =
            string::tokenize(props.string("bins"), " ,");

        for (const std::string &token : tokens) {
            std::vector<std::string> item = string::tokenize(token, ":");

            if (item.size() != 3 || item[0].empty() || item[1].empty() ||
                item[2].empty()) {
                Log(Warn, "Invalid spectral bin specification '%s', skipping",
                    token);
                continue;
            }
            m_bin_names.push_back(item[0]);

            ScalarFloat lower;

            try {
                lower = (ScalarFloat) std::stod(item[1]);
            } catch (...) {
                Throw("Could not parse floating point value '%s'", item[1]);
            }
            m_bin_lower_bounds.push_back(lower);

            ScalarFloat upper;
            try {
                upper = (ScalarFloat) std::stod(item[2]);
            } catch (...) {
                Throw("Could not parse floating point value '%s'", item[2]);
            }
            m_bin_upper_bounds.push_back(upper);
        }

        if (m_bin_names.empty())
            Log(Warn, "No spectral bin was specified!");

        // Final pre-processing steps
        PluginManager *pmgr = PluginManager::instance();

        for (size_t i = 0; i < m_bin_names.size(); ++i) {
            Properties props("uniform");
            props.set_float("lambda_min", m_bin_lower_bounds[i]);
            props.set_float("lambda_max", m_bin_upper_bounds[i]);
            props.set_float("value", 1.0);
            m_bin_weights.push_back(pmgr->create_object<Texture>(props));
        }
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
            SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
            si.wavelengths          = ray.wavelengths;

            for (size_t i = 0; i < m_bin_names.size(); ++i) {
                auto bin_weights = m_bin_weights[i]->eval(si, active);
                *aovs++          = hsum(bin_weights * result.first);
                *aovs++          = hsum(bin_weights);
            }
        }

        // Forward result
        return result;
    }

    std::vector<std::string> aov_names() const override {
        std::vector<std::string> names;
        for (auto &bin_name : m_bin_names) {
            names.push_back(bin_name);
            names.push_back(bin_name + "_weights");
        }
        return names;
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("integrator", m_integrator.get());
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "BinsIntegrator[" << std::endl
            << "  bin_names = " << string::indent(m_bin_names) << ","
            << std::endl
            << "  bin_weights = " << string::indent(m_bin_weights) << ","
            << std::endl
            << "  integrator = " << string::indent(m_integrator) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    std::vector<std::string> m_bin_names;
    std::vector<ScalarFloat> m_bin_lower_bounds;
    std::vector<ScalarFloat> m_bin_upper_bounds;
    ref<Base> m_integrator;
    std::vector<ref<Texture>> m_bin_weights;
};

MTS_IMPLEMENT_CLASS_VARIANT(BinsIntegrator, SamplingIntegrator)
MTS_EXPORT_PLUGIN(BinsIntegrator, "Bins integrator");
NAMESPACE_END(mitsuba)
