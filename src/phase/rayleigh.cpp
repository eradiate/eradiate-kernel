#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/distr_1d.h>
#include <mitsuba/render/phase.h>

/**!

.. _phase-rayleigh:

Rayleigh phase function (:monosp:`rayleigh`)
-----------------------------------------------

.. list-table::
 :widths: 20 15 65
 :header-rows: 1
 :class: paramstable

 * - Parameter
   - Type
   - Description
 * - delta
   - |float|
   - This parameter specifies the particle depolarization factor
     (usually between 0 and 0.5, can take values in :math:`[0,1[`). Default: 0

This plugin implements the scalar version of the Rayleigh phase function model
proposed by |nbsp| :cite:`Hansen1974a`. It is parametrizable by the species 
depolarization factor :math:`\delta`.

*/

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class RayleighPhaseFunction final : public PhaseFunction<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(PhaseFunction, m_flags)
    MTS_IMPORT_TYPES(PhaseFunctionContext)

    using FloatStorage = DynamicBuffer<Float>;

    RayleighPhaseFunction(const Properties &props) : Base(props) {
        if constexpr (is_polarized_v<Spectrum>)
            Log(
                Warn, 
                "Polarized version of Rayleigh phase function not implemented, "
                "falling back to scalar version"
            );

        m_delta = props.float_("delta", 0.f);
        if (m_delta < 0 || m_delta >= 1)
            Log(
                Error, 
                "The depolarization factor must lie in the interval (0, 1(!"
            );
        m_flags = +PhaseFunctionFlags::Anisotropic;
        update();
    }

    void update() {
        m_ddelta = m_delta != 0.f ? (1.f - m_delta) / (1.f + 0.5f * m_delta)
                                  : 1.f;
        m_a = 3.f * m_ddelta / (16.f * math::Pi<ScalarFloat>);
        m_b = (1.f - m_ddelta) / (4.f * math::Pi<ScalarFloat>);

        ScalarFloat cos_theta_min = -1.f,
                    cos_theta_max = 1.f;
        int n_cos_theta = int(201);
        auto cos_theta = linspace<FloatStorage>(cos_theta_min, cos_theta_max, n_cos_theta);

        auto phase_func_values = zero<FloatStorage>(n_cos_theta);
        for (int i = 0; i != n_cos_theta; ++i)
            phase_func_values[i] = eval_rayleigh_scalar(cos_theta[i]);

        m_table.range() = ScalarVector2f(cos_theta_min, cos_theta_max);
        m_table.pdf() = phase_func_values;
        m_table.update();
    }

    ScalarFloat eval_rayleigh_scalar(ScalarFloat cos_theta) const {
        return m_a * (1.f + cos_theta * cos_theta) + m_b;
    }

    MTS_INLINE Float eval_rayleigh(Float cos_theta) const {
        auto result = m_a * (1.f + enoki::sqr(cos_theta)) + m_b;
        return result;
    }

    std::pair<Vector3f, Float> sample(const PhaseFunctionContext & /* ctx */,
                                      const MediumInteraction3f &mi, 
                                      const Point2f &sample,
                                      Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        Float cos_theta = m_table.sample(sample.x());
        Float sin_theta = enoki::safe_sqrt(1.0f - cos_theta * cos_theta);

        auto [sin_phi, cos_phi] = enoki::sincos(2 * math::Pi<ScalarFloat> * sample.y());
        auto wo = Vector3f(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
        wo = mi.to_world(wo);
        Float pdf = eval_rayleigh(cos_theta);
        return std::make_pair(wo, pdf);
    }

    Float eval(const PhaseFunctionContext & /* ctx */, const MediumInteraction3f &mi,
               const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);
        return eval_rayleigh(dot(wo, mi.wi));
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("delta", m_delta);
        update();
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "RayleighPhaseFunction[" << std::endl
            << "  delta = " << string::indent(m_delta) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()

private:
    ScalarFloat m_delta;
    ScalarFloat m_ddelta;
    ScalarFloat m_a;
    ScalarFloat m_b;
    ContinuousDistribution<Float> m_table;
};


MTS_IMPLEMENT_CLASS_VARIANT(RayleighPhaseFunction, PhaseFunction)
MTS_EXPORT_PLUGIN(RayleighPhaseFunction, "Rayleigh phase function")
NAMESPACE_END(mitsuba)