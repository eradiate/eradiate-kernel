import numpy as np
import pytest

import enoki as ek
import mitsuba


def test_create_rpv3(variant_scalar_rgb):
    # Test constructor of 3-parameter version of RPV
    from mitsuba.core.xml import load_string
    from mitsuba.render import BSDFFlags

    rpv = load_string("<bsdf version='2.0.0' type='rpv'/>")
    assert rpv is not None
    assert rpv.component_count() == 1
    assert rpv.flags(0) == BSDFFlags.GlossyReflection | BSDFFlags.FrontSide
    assert rpv.flags() == rpv.flags(0)


def test_chi2_rpv3(variant_packet_rgb):
    from mitsuba.python.chi2 import BSDFAdapter, ChiSquareTest, SphericalDomain

    sample_func, pdf_func = BSDFAdapter("rpv", "")

    chi2 = ChiSquareTest(
        domain=SphericalDomain(),
        sample_func=sample_func,
        pdf_func=pdf_func,
        sample_dim=3,
    )

    assert chi2.run()


def rpv_reference(rho_0, rho_0_hotspot, theta, k,
                  theta_i, phi_i, theta_o, phi_o):
    """Reference for RPV, adapted from a C implementation."""

    sini, ui = ek.sincos(theta_i)
    tan_i = sini / ui
    sino, uo = ek.sincos(theta_o)
    tan_o = sino / uo
    cosphi = ek.cos(phi_i - phi_o)

    K1 = ek.pow(ui * uo * (ui + uo), k - 1.)

    cos_g = ui * uo + sini * sino * cosphi

    FgDenum = 1. + theta * theta + 2. * theta * cos_g
    Fg = (1. - theta * theta) / ek.pow(FgDenum, 1.5)

    G = ek.sqrt(tan_i * tan_i + tan_o * tan_o - 2. * tan_i * tan_o * cosphi)
    K3 = 1. + (1. - rho_0_hotspot) / (1. + G)

    return (K1 * Fg * K3 * rho_0 * ek.abs(uo))


@pytest.mark.parametrize("rho_0", [0.1, 0.497, 0.004])
@pytest.mark.parametrize("k", [0.543, 0.851, 0.634])
@pytest.mark.parametrize("theta", [-0.29, 0.086, 0.2])
def test_eval(variant_scalar_rgb, rho_0, k, theta):
    """Test the eval method of the RPV plugin, comparing to a reference 
    implementation."""

    from mitsuba.core.xml import load_string
    from mitsuba.core import Vector3f
    from mitsuba.render import BSDFContext, SurfaceInteraction3f

    rpv = load_string(f"""
    <bsdf version="2.0.0" type="rpv">
        <float name="rho_0" value="{rho_0}"/>
        <float name="k" value="{k}"/>
        <float name="ttheta" value="{theta}"/>
    </bsdf>""")

    num_samples = 100

    theta_i = np.random.rand(num_samples) * np.pi / 2.
    theta_o = np.random.rand(num_samples) * np.pi / 2.
    phi_i = np.random.rand(num_samples) * np.pi * 2.
    phi_o = np.random.rand(num_samples) * np.pi * 2.

    value = []
    reference = []
    for i in range(num_samples):
        ti = theta_i[i]
        to = theta_o[i]
        pi = phi_i[i]
        po = phi_o[i]

        wi = Vector3f(ek.sin(ti) * ek.cos(pi),
                      ek.sin(ti) * ek.sin(pi), ek.cos(ti))
        wo = Vector3f(ek.sin(to) * ek.cos(po),
                      ek.sin(to) * ek.sin(po), ek.cos(to))

        si = SurfaceInteraction3f()
        si.wi = wi
        ctx = BSDFContext()

        value.append(rpv.eval(ctx, si, wo, True)[0])

        reference.append(rpv_reference(rho_0, rho_0, theta, k, ti, pi, to, po))

    assert ek.allclose(value, reference, rtol=1e-3, atol=1e-3)
