import mitsuba


def test_create(variant_scalar_rgb):
    from mitsuba.core.xml import load_string
    p = load_string("""<phase version='2.0.0' type='rayleigh'/>""")
    assert p is not None


def test_create(variant_scalar_mono_polarized):
    from mitsuba.core.xml import load_string
    p = load_string("""<phase version='2.0.0' type='rayleigh'/>""")
    assert p is not None


def test_chi2(variant_packet_rgb):
    from mitsuba.python.chi2 import PhaseFunctionAdapter, ChiSquareTest, SphericalDomain

    sample_func, pdf_func = PhaseFunctionAdapter(
        "rayleigh", '<float name="delta" value="0."/>'
    )

    chi2 = ChiSquareTest(
        domain=SphericalDomain(),
        sample_func=sample_func,
        pdf_func=pdf_func,
        sample_dim=2,
        sample_count=int(1e6),
    )

    result = chi2.run()
    chi2._dump_tables()
    assert result
