#include <mitsuba/core/frame.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _bsdf-rpv:

Rahman Pinty Verstraete reflection model (:monosp:`RPV`)
--------------------------------------------------------

.. pluginparameters::

 * - rho_0
   - |spectrum| or |texture|
   - :math:`\rho_0 \ge 0`. Default: 0.1
 * - k
   - |spectrum| or |texture|
   - :math:`k \in \mathbb{R}`. Default: 0.1
 * - ttheta
   - |spectrum| or |texture|
   - :math:`-1 \le \Theta \le 1`. Default: 0.0
 * - rho_c
   - |spectrum| or |texture|
   - Default: Equal to rho_0
 * - sigma
   - |spectrum| or |texture|
   - Default: 0.0
 * - delta
   - |spectrum| or |texture|
   - Default: 1.0

This plugin implements the reflection model proposed by
:cite:`Rahman1993CoupledSurfaceatmosphereReflectance`.

Apart from homogeneous values, the plugin can also accept
nested or referenced texture maps to be used as the source of parameter
information, which is then mapped onto the shape based on its UV
parameterization. When no parameters are specified, the model uses the default
values of :math:`\rho_0 = 0.1`, :math:`k = 0.1` and :math:`\Theta = 0.0`

The model includes several extensions, which introduce up to three optional
parameters. The following formulae demonstrate these extensions. For the
fundamental formulae defining the RPV model please refer to the Eradiate
Scientific Handbook.

- Parameter :math:`\rho_c`:

  :math:`1+R(G)=1+\frac{1-\rho_c}{1+G}`

  If not given, it defaults to :math:`\rho_0`, yielding the fundamental
  version of the formula.
- Parameter :math:`\delta`:

  :math:`1 + R(G) = 1 + \frac{1 - \rho_0}{\delta + G}`

  If not given, it defaults to :math:`1`, yielding the fundamental version of
  the formula.
- Parameter :math:`\sigma`:

  .. math::

     \rho_s = \rho_0
        \left[
            \frac{
              \cos^{k-1} \theta_1 \cos^{k-1} \theta_2
            }{
              (\cos \theta_1 + \cos \theta_2)^{1-k}
            } \cdot F(g) [1 + R(G)] + \frac{\sigma}{\cos \theta_1}
        \right]

  If not given, it defaults to :math:`0`, yielding the fundamental version
  of the formula.

Note that this material is one-sided, that is, observed from the
back side, it will be completely black. If this is undesirable,
consider using the :ref:`twosided <bsdf-twosided>` BRDF adapter plugin.
The following XML snippet describes an RPV material with monochromatic
parameters:

.. code-block:: xml
    :name: rpv-monochrome

    <bsdf type="rpv">
        <float name="rho_0" value="0.02"/>
        <float name="k" value="0.3"/>
        <float name="ttheta" value="-0.12"/>
    </bsdf>

*/

MTS_VARIANT
class RPV final : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, m_flags, m_components)
    MTS_IMPORT_TYPES(Texture)

    RPV(const Properties &props) : Base(props) {
        m_rho_0  = props.texture<Texture>("rho_0", 0.1f);
        m_ttheta = props.texture<Texture>("ttheta", 0.f);
        m_k      = props.texture<Texture>("k", 0.1f);
        m_sigma  = props.texture<Texture>("sigma", 0.0f);
        m_delta  = props.texture<Texture>("delta", 1.0f);
        if (props.has_property("rho_c")) {
            m_rho_c = props.texture<Texture>("rho_c", 0.1f);
        } else {
            m_rho_c = m_rho_0;
        }
        m_flags = BSDFFlags::GlossyReflection | BSDFFlags::FrontSide;
        m_components.push_back(m_flags);
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext & /* ctx */,
                                             const SurfaceInteraction3f &si,
                                             Float /* position_sample */,
                                             const Point2f &direction_sample,
                                             Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        BSDFSample3f bs   = zero<BSDFSample3f>();

        active &= cos_theta_i > 0.f;

        bs.wo           = warp::square_to_cosine_hemisphere(direction_sample);
        bs.pdf          = warp::square_to_cosine_hemisphere_pdf(bs.wo);
        bs.eta          = 1.f;
        bs.sampled_type = +BSDFFlags::GlossyReflection;

        Spectrum value = eval_rpv(si, bs.wo, active);
        return { bs, select(active && bs.pdf > 0.f,
                            unpolarized<Spectrum>(value), 0.f) };
    }

    Spectrum eval_rpv(const SurfaceInteraction3f &si, const Vector3f &wo,
                      Mask active) const {
        Spectrum rho_0  = m_rho_0->eval(si, active);
        Spectrum rho_c  = m_rho_c->eval(si, active);
        Spectrum ttheta = m_ttheta->eval(si, active);
        Spectrum k      = m_k->eval(si, active);
        Spectrum delta  = m_delta->eval(si, active);
        Spectrum sigma  = m_sigma->eval(si, active);

        auto [sin_phi1, cos_phi1] = Frame3f::sincos_phi(si.wi);
        auto [sin_phi2, cos_phi2] = Frame3f::sincos_phi(wo);
        Float cos_phi1_minus_phi2 = cos_phi1 * cos_phi2 + sin_phi1 * sin_phi2;
        Float sin_theta1          = Frame3f::sin_theta(si.wi);
        Float cos_theta1          = Frame3f::cos_theta(si.wi);
        Float tan_theta1          = Frame3f::tan_theta(si.wi);
        Float sin_theta2          = Frame3f::sin_theta(wo);
        Float cos_theta2          = Frame3f::cos_theta(wo);
        Float tan_theta2          = Frame3f::tan_theta(wo);

        Float G =
            safe_sqrt(sqr(tan_theta1) + sqr(tan_theta2) -
                      2.f * tan_theta1 * tan_theta2 * cos_phi1_minus_phi2);
        Float cos_g = cos_theta1 * cos_theta2 +
                      sin_theta1 * sin_theta2 * cos_phi1_minus_phi2;
        // the following uses cos(\pi-x) = -cos(x)
        Spectrum F = (1.f - sqr(ttheta)) /
                     pow((1.f + sqr(ttheta) + 2.f * ttheta * cos_g), 1.5f);

        Spectrum value =
            rho_0 *
            (pow(cos_theta1 * cos_theta2 * (cos_theta1 + cos_theta2), k - 1.f) *
                 F * (1.f + (1.f - rho_c) / (delta + G)) +
             sigma / cos_theta1);

        return value;
    }

    Spectrum eval(const BSDFContext & /*ctx*/, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);
        Spectrum value = eval_rpv(si, wo, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        return select(active, unpolarized<Spectrum>(value) * abs(cos_theta_o),
                      0.f);
    }

    Float pdf(const BSDFContext & /* ctx */, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        Float pdf = warp::square_to_cosine_hemisphere_pdf(wo);

        return select(cos_theta_i > 0.f && cos_theta_o > 0.f, pdf, 0.f);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("rho_0", m_rho_0.get());
        callback->put_object("ttheta", m_ttheta.get());
        callback->put_object("k", m_k.get());
        callback->put_object("rho_c", m_rho_c.get());
        callback->put_object("delta", m_delta.get());
        callback->put_object("sigma", m_sigma.get());
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "RPV[" << std::endl
            << "  rho_0 = " << string::indent(m_rho_0) << std::endl
            << "  ttheta = " << string::indent(m_ttheta) << std::endl
            << "  k = " << string::indent(m_k) << std::endl
            << "  rho_c = " << string::indent(m_rho_c) << std::endl
            << "  delta = " << string::indent(m_delta) << std::endl
            << "  sigma = " << string::indent(m_sigma) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_rho_0;
    ref<Texture> m_ttheta;
    ref<Texture> m_k;
    ref<Texture> m_rho_c;
    ref<Texture> m_sigma;
    ref<Texture> m_delta;
};

MTS_IMPLEMENT_CLASS_VARIANT(RPV, BSDF)
MTS_EXPORT_PLUGIN(RPV, "Rahman-Pinty-Verstraete BSDF")
NAMESPACE_END(mitsuba)