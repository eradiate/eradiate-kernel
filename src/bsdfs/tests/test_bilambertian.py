import mitsuba
import pytest
import enoki as ek
import numpy as np


def test_instantiation(variant_scalar_rgb):
    from mitsuba.render import BSDFFlags
    from mitsuba.core.xml import load_dict

    b = load_dict({"type": "bilambertian"})
    assert b is not None
    assert b.component_count() == 2
    expected_flags_reflection = \
        BSDFFlags.DiffuseReflection | BSDFFlags.FrontSide | BSDFFlags.BackSide
    expected_flags_transmission = \
        BSDFFlags.DiffuseTransmission | BSDFFlags.FrontSide | BSDFFlags.BackSide
    assert b.flags(0) == expected_flags_reflection
    assert b.flags(1) == expected_flags_transmission


def test_eval_pdf(variant_scalar_rgb):
    from mitsuba.core.xml import load_dict
    from mitsuba.render import BSDFContext, SurfaceInteraction3f
    from mitsuba.core import Frame3f

    for (r, t) in [(0.2, 0.4), (0.4, 0.2),
                   (0.1, 0.9), (0.9, 0.1),
                   (0.4, 0.6), (0.6, 0.4)]:
        albedo = r + t

        bsdf = load_dict({
            "type": "bilambertian",
            "reflectance": r,
            "transmittance": t
        })

        ctx = BSDFContext()

        si = SurfaceInteraction3f()
        si.p = [0, 0, 0]
        si.n = [0, 0, 1]

        for wi in [[0, 0, 1], [0, 0, -1]]:  # We try from both the front and back sides
            si.wi = wi
            si.sh_frame = Frame3f(si.n)

            for i in range(20):
                theta = i / 19.0 * ek.pi  # We cover the entire circle

                wo = [ek.sin(theta), 0, ek.cos(theta)]
                v_pdf = bsdf.pdf(ctx, si, wo=wo)
                v_eval = bsdf.eval(ctx, si, wo=wo)

                if ek.dot(wi, wo) > 0:
                    # reflection
                    assert ek.allclose(v_eval, r * ek.abs(wo[2]) / ek.pi)
                    assert ek.allclose(v_pdf, r / albedo *
                                       ek.abs(wo[2]) / ek.pi)
                else:
                    # transmission
                    assert ek.allclose(v_eval, t * ek.abs(wo[2]) / ek.pi)
                    assert ek.allclose(v_pdf, t / albedo *
                                       ek.abs(wo[2]) / ek.pi)


@pytest.mark.parametrize("r,t", [
    [0.6, 0.2],
    [0.2, 0.6],
    [0.6, 0.4],
    [0.4, 0.6],
    [0.9, 0.1],
    [0.1, 0.9],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
])
def test_chi2(variant_packet_rgb, r, t):
    from mitsuba.python.chi2 import BSDFAdapter, ChiSquareTest, SphericalDomain

    xml = f"""
        <spectrum name="reflectance" value="{r}"/>
        <spectrum name="transmittance" value="{t}"/>
    """

    sample_func, pdf_func = BSDFAdapter("bilambertian", xml)

    chi2 = ChiSquareTest(
        domain=SphericalDomain(),
        sample_func=sample_func,
        pdf_func=pdf_func,
        sample_dim=3
    )

    assert chi2.run()
