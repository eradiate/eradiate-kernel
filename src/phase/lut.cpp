#include <mitsuba/core/distr_1d.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/phase.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _phase-lut:

Lookup table phase function (:monosp:`lut`)
-------------------------------------------

.. pluginparameters::

 * - values
   - |string|
   - A comma-separated list of phase function values parametrised by the
     cosine of the scattering angle.

This plugin implements a generic phase function model for isotropic media
parametrised by a lookup  table giving values of the phase function as a
function of the cosine of the scattering angle.

.. admonition:: Notes

   * The scattering angle is here defined as the dot product of the
     incoming and outgoing directions, where the incoming direction points
     *toward* the interaction point.
   * Lookup table points are regularly spaced between -1 and 1.
   * Phase function values are automatically normalized.
*/

template <typename Float, typename Spectrum>
class LUTPhaseFunction final : public PhaseFunction<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(PhaseFunction, m_flags, m_components)
    MTS_IMPORT_TYPES(PhaseFunctionContext)

    LUTPhaseFunction(const Properties &props) : Base(props) {
        if (props.type("values") == Properties::Type::String) {
            std::vector<std::string> values_str =
                string::tokenize(props.string("values"), " ,");
            std::vector<ScalarFloat> data;
            data.reserve(values_str.size());

            for (const auto &s : values_str) {
                try {
                    data.push_back((ScalarFloat) std::stod(s));
                } catch (...) {
                    Throw("Could not parse floating point value '%s'", s);
                }
            }

            m_distr = ContinuousDistribution<Float>(ScalarVector2f(-1.f, 1.f),
                                                    data.data(), data.size());
        } else {
            Throw("'values' must be a string");
        }

        m_flags = +PhaseFunctionFlags::Anisotropic;
        m_components.push_back(m_flags);
    }

    std::pair<Vector3f, Float> sample(const PhaseFunctionContext & /* ctx */,
                                      const MediumInteraction3f &mi,
                                      Float /* sample1 */,
                                      const Point2f &sample2,
                                      Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        Float cos_theta = m_distr.sample(sample2.x());
        Float sin_theta = enoki::safe_sqrt(1.0f - cos_theta * cos_theta);
        auto [sin_phi, cos_phi] =
            enoki::sincos(2.f * math::Pi<ScalarFloat> * sample2.y());
        auto wo = Vector3f(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
        wo      = mi.to_world(wo);
        Float pdf = m_distr.eval_pdf_normalized(-cos_theta, active) *
                    math::InvTwoPi<ScalarFloat>;

        return std::make_pair(wo, pdf);
    }

    Float eval(const PhaseFunctionContext & /* ctx */,
               const MediumInteraction3f &mi, const Vector3f &wo,
               Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);

        Float cos_theta = dot(wo, mi.wi);
        return m_distr.eval_pdf_normalized(-cos_theta, active) *
               math::InvTwoPi<ScalarFloat>;
    }

    void traverse(TraversalCallback * /* callback */) override {
        NotImplementedError("traverse");
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "LUTPhaseFunction[" << std::endl
            << "  distr = " << string::indent(m_distr) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ContinuousDistribution<Float> m_distr;
};

MTS_IMPLEMENT_CLASS_VARIANT(LUTPhaseFunction, PhaseFunction)
MTS_EXPORT_PLUGIN(LUTPhaseFunction, "Lookup table-based phase function")
NAMESPACE_END(mitsuba)
