#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/fresnel.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _bsdf-bilambertian:

Bi-Lambertian material (:monosp:`bilambertian`)
-----------------------------------------------

.. pluginparameters::

 * - reflectance
   - |spectrum| or |texture|
   - Specifies the diffuse reflectance of the material (Default: 0.5)
 * - transmittance
   - |spectrum| or |texture|
   - Specifies the diffuse transmittance of the material (Default: 0.5)

The bi-Lambertian material represents a material that scatters light diffusely
into the entire sphere. The reflectance specifies the amount of light scattered
into the incoming hemisphere, while the transmittance specifies the amount of 
light scattered into the outgoing hemisphere. This material is two-sided.

.. subfigstart::
.. subfigure:: ../../resources/data-extra/docs/images/render/bsdf_bilambertian_reflective.jpg
   :caption: With dominant reflectivity
.. subfigure:: ../../resources/data-extra/docs/images/render/bsdf_bilambertian_transmissive.jpg
   :caption: With dominant transmissivity
.. subfigend::
   :label: fig-bilambertian

*/

template <typename Float, typename Spectrum>
class BiLambertian final : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, m_flags, m_components)
    MTS_IMPORT_TYPES(Texture)

    BiLambertian(const Properties &props) : Base(props) {
        m_reflectance   = props.texture<Texture>("reflectance", .5f);
        m_transmittance = props.texture<Texture>("transmittance", .5f);

        m_components.push_back(BSDFFlags::DiffuseReflection |
                               BSDFFlags::FrontSide | BSDFFlags::BackSide);
        m_components.push_back(BSDFFlags::DiffuseTransmission |
                               BSDFFlags::FrontSide | BSDFFlags::BackSide);

        m_flags = m_components[0] | m_components[1];
    }

    std::pair<BSDFSample3f, Spectrum>
    sample(const BSDFContext &ctx, const SurfaceInteraction3f &si,
           Float sample1, const Point2f &sample2, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        bool has_reflect  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 0),
             has_transmit = ctx.is_enabled(BSDFFlags::DiffuseTransmission, 1);

        if (unlikely((!has_reflect && !has_transmit) || none_or<false>(active)))
            return { zero<BSDFSample3f>(), UnpolarizedSpectrum(0.f) };

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        Vector3f wo       = warp::square_to_cosine_hemisphere(sample2);
        Float cos_theta_o = Frame3f::cos_theta(wo);

        BSDFSample3f bs   = zero<BSDFSample3f>();
        UnpolarizedSpectrum value(0.f);

        // Select the lobe to be sampled
        UnpolarizedSpectrum r            = m_reflectance->eval(si, active),
                            t            = m_transmittance->eval(si, active);
        Float reflection_sampling_weight = hmean(r / (r + t));
        // Handle case where r = t = 0
        masked(reflection_sampling_weight, isnan(reflection_sampling_weight)) = 0.f;
        
        Mask selected_r = (sample1 < reflection_sampling_weight) && active,
             selected_t = (sample1 >= reflection_sampling_weight) && active;

        // Evaluate
        value = select(active, Float(1.f), 0.f);
        value[selected_r] *= r;
        value[selected_t] *= t;

        // Compute PDF
        bs.pdf = select(active, warp::square_to_cosine_hemisphere_pdf(wo), 0.f);
        bs.pdf = select(selected_r, bs.pdf * reflection_sampling_weight, bs.pdf);
        bs.pdf = select(selected_t, bs.pdf * (1.f - reflection_sampling_weight), 
                        bs.pdf);

        // Set other interaction fields
        bs.eta               = 1.f;
        bs.sampled_component = select(selected_r, UInt32(0), UInt32(1));
        bs.sampled_type =
            select(selected_r, UInt32(+BSDFFlags::DiffuseReflection),
                   UInt32(+BSDFFlags::DiffuseTransmission));

        // Flip the outgoing direction if the incoming comes from "behind"
        wo = select(cos_theta_i > 0, wo, Vector3f(wo.x(), wo.y(), -wo.z()));
        
        // Flip the outgoing direction if transmission was selected
        bs.wo = select(selected_r, wo, Vector3f(wo.x(), wo.y(), -wo.z()));

        return { bs, select(active && bs.pdf > 0.f,
                            unpolarized<Spectrum>(value), 0.f) };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_reflect  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 0),
             has_transmit = ctx.is_enabled(BSDFFlags::DiffuseTransmission, 1);

        if (unlikely((!has_reflect && !has_transmit) || none_or<false>(active)))
            return 0.f;

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        UnpolarizedSpectrum result(0.f);

        if (has_reflect) {
            Mask is_reflect =
                Mask(sign(cos_theta_i) == sign(cos_theta_o)) && active;
            result[is_reflect] = m_reflectance->eval(si, is_reflect);
        }

        if (has_transmit) {
            Mask is_transmit =
                Mask(sign(cos_theta_i) != sign(cos_theta_o)) && active;
            result[is_transmit] = m_transmittance->eval(si, is_transmit);
        }

        result[active] *= math::InvPi<Float> * abs(cos_theta_o);
        return select(active, result, 0.f);
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_reflect  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 0),
             has_transmit = ctx.is_enabled(BSDFFlags::DiffuseTransmission, 1);

        if (unlikely((!has_reflect && !has_transmit) || none_or<false>(active)))
            return 0.f;

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);
        Vector3f wo_flip{ wo.x(), wo.y(), abs(cos_theta_o) };
        Float result =
            select(active, warp::square_to_cosine_hemisphere_pdf(wo_flip), 0.f);

        UnpolarizedSpectrum r            = m_reflectance->eval(si, active),
                            t            = m_transmittance->eval(si, active);
        Float reflection_sampling_weight = hmean(r / (r + t));
        // Handle case where r = t = 0
        masked(reflection_sampling_weight, isnan(reflection_sampling_weight)) = 0.f;
        
        if (has_reflect) {
            Mask is_reflect =
                Mask(sign(cos_theta_i) == sign(cos_theta_o)) && active;
            masked(result, is_reflect) *= reflection_sampling_weight;
        }

        if (has_transmit) {
            Mask is_transmit =
                Mask(sign(cos_theta_i) != sign(cos_theta_o)) && active;
            masked(result, is_transmit) *= (1.f - reflection_sampling_weight);
        }

        return result;
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("reflectance", m_reflectance.get());
        callback->put_object("transmittance", m_transmittance.get());
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Bilambertian[" << std::endl
            << "  reflectance = " << string::indent(m_reflectance) << std::endl
            << "  transmittance = " << string::indent(m_transmittance)
            << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_reflectance;
    ref<Texture> m_transmittance;
};

MTS_IMPLEMENT_CLASS_VARIANT(BiLambertian, BSDF)
MTS_EXPORT_PLUGIN(BiLambertian, "Bi-Lambertian material")
NAMESPACE_END(mitsuba)
