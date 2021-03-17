import numpy as np
import mitsuba

def test_create(variant_scalar_rgb):
    from mitsuba.core.xml import load_dict
    p = load_dict({"type": "lut", "values": "2, 1, 2"})
    assert p is not None


def test_eval(variant_scalar_rgb):
    from mitsuba.core.math import Pi
    from mitsuba.core.xml import load_dict
    from mitsuba.render import PhaseFunctionContext, MediumInteraction3f

    iso = load_dict({"type": "isotropic"})
    lut = load_dict({"type": "lut", "values": "1, 1, 1"})
    ctx = PhaseFunctionContext(None)
    mi = MediumInteraction3f()
    for theta in np.linspace(0, np.pi / 2, 4):
        for ph in np.linspace(0, np.pi, 4):
            wo = [np.sin(theta), 0, np.cos(theta)]
            iso_eval = iso.eval(ctx, mi, wo)
            lut_eval = lut.eval(ctx, mi, wo)
            assert np.allclose(iso_eval, lut_eval)

def test_chi2(variant_packet_rgb):
    from mitsuba.python.chi2 import PhaseFunctionAdapter, ChiSquareTest, SphericalDomain

    sample_func, pdf_func = PhaseFunctionAdapter("lut", '<string name="values" value="2, 1, 2"/>')

    chi2 = ChiSquareTest(
        domain = SphericalDomain(),
        sample_func = sample_func,
        pdf_func = pdf_func,
        sample_dim = 2
    )

    result = chi2.run(0.1)
    chi2._dump_tables()
    assert result
